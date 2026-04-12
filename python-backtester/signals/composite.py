"""Composite Signal: AND-gates VRP and Regime, applies supplementary adjustments."""

import logging
from datetime import date
import math
from typing import Optional

from config.models import SignalsConfig
from data.manager import DataManager
from signals.base import SignalOutput, TradeDecision
from signals.vrp_signal import VRPSignal
from signals.regime_signal import ForecastUncertaintyRegime

logger = logging.getLogger(__name__)


class CompositeSignal:
    """
    AND-gates Signal 1 (VRP) and Signal 2 (Regime).
    Applies supplementary adjustments from classification models.
    """

    def __init__(self, config: SignalsConfig, data_manager: DataManager):
        self._config = config
        self._dm = data_manager
        self._vrp = VRPSignal(config.vrp, data_manager)
        self._regime = ForecastUncertaintyRegime(config.regime, data_manager)

    def precompute(self, start: date, end: date) -> None:
        """Precompute both signal series."""
        logger.info("Precomputing VRP signal...")
        self._vrp.precompute(start, end)
        logger.info("Precomputing regime signal...")
        self._regime.precompute(start, end)

    def evaluate(self, dt: date) -> TradeDecision:
        """Evaluate composite signal and return a trade decision."""
        vrp_out = self._vrp.compute(dt)
        regime_out = self._regime.compute(dt)

        reason_codes = []

        # If either signal is unavailable, no trade
        if vrp_out is None:
            reason_codes.append("NO_VRP_DATA")
            return self._flat(dt, vrp_out, regime_out, reason_codes)

        if regime_out is None:
            reason_codes.append("NO_REGIME_DATA")
            return self._flat(dt, vrp_out, regime_out, reason_codes)

        # AND-gate: both must permit
        s1_scalar = vrp_out.size_scalar
        s2_multiplier = regime_out.size_scalar
        vrp_raw = vrp_out.raw_value

        trade_permitted = (vrp_raw > 0) and (s1_scalar > 0) and (s2_multiplier > 0)

        if vrp_raw <= 0:
            reason_codes.append("VRP_NEGATIVE")
        if s1_scalar == 0:
            reason_codes.append(f"VRP_INVERTED (z={vrp_out.z_score:.2f})")
        if s2_multiplier == 0:
            reason_codes.append(f"REGIME_CRISIS (z={regime_out.z_score:.2f})")

        if not trade_permitted:
            return self._flat(dt, vrp_out, regime_out, reason_codes)

        # Composite scalar: conservative sizing via sqrt(1 - product)
        raw_scalar = s1_scalar * s2_multiplier
        if raw_scalar < 1.0:
            final_scalar = math.sqrt(1.0 - raw_scalar)
        else:
            final_scalar = 0.0

        # Supplementary adjustments from classification models
        supp_config = self._config.supplementary

        # VIX classification: if VIX expected to rise with high probability, reduce
        vix_class = self._dm.get_vix_classification_row(dt)
        if vix_class is not None:
            if (vix_class.get("pred_class_optimal", 0) == 1 and
                    vix_class.get("pred_prob", 0) > supp_config.vix_class_prob_threshold):
                reduction = supp_config.supplementary_reduction
                final_scalar *= (1.0 - reduction)
                reason_codes.append(f"VIX_CLASS_REDUCE_{reduction:.0%}")

        # RV classification: if RV expected to rise with high probability, reduce
        rv_class = self._dm.get_rv_classification_row(dt)
        if rv_class is not None:
            if (rv_class.get("pred_class_optimal", 0) == 1 and
                    rv_class.get("pred_prob", 0) > supp_config.rv_class_prob_threshold):
                reduction = supp_config.supplementary_reduction
                final_scalar *= (1.0 - reduction)
                reason_codes.append(f"RV_CLASS_REDUCE_{reduction:.0%}")

        # Determine if call overlay eligible
        call_overlay = (vrp_out.regime == "EXTREMELY_RICH" and regime_out.regime == "STABLE")
        if call_overlay:
            reason_codes.append("CALL_OVERLAY_ELIGIBLE")

        # Binding signal
        if s1_scalar < s2_multiplier:
            reason_codes.append("BINDING: VRP")
        elif s2_multiplier < s1_scalar:
            reason_codes.append("BINDING: REGIME")
        else:
            reason_codes.append("BINDING: EQUAL")

        reason_codes.append(f"ENTER: scalar={final_scalar:.3f}")

        return TradeDecision(
            date=dt,
            action="ENTER",
            size_scalar=final_scalar,
            signals={"vrp": vrp_out, "regime": regime_out},
            reason_codes=reason_codes,
        )

    def _flat(
        self,
        dt: date,
        vrp_out: Optional[SignalOutput],
        regime_out: Optional[SignalOutput],
        reason_codes: list,
    ) -> TradeDecision:
        return TradeDecision(
            date=dt,
            action="FLAT",
            size_scalar=0.0,
            signals={"vrp": vrp_out, "regime": regime_out},
            reason_codes=reason_codes,
        )
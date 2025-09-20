"""Utilities for exploring Swedish income tax and pension contributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TaxParameters:
    """Container for the 2025 Stockholm tax and pension settings."""

    pbb: float = 58_800.0
    ibb: float = 80_600.0
    municipal_rate: float = 0.3060
    state_rate: float = 0.20
    skiktgrans: float = 625_800.0
    brytpunkt: float = 643_100.0
    burial_fee: float = 0.00070
    public_service_max: float = 1_249.0
    public_service_rate: float = 0.01
    public_pension_rate: float = 0.185
    pgi_rate: float = 0.93
    pgi_ceiling: float = 7.5 * ibb
    btp1_rate_low: float = 0.065
    btp1_rate_high: float = 0.32
    btp1_low_ceiling: float = 7.5 * ibb
    btp1_high_ceiling: float = 30.0 * ibb
    fee_cap: float = 8.07 * ibb

    months_per_year: int = 12


def grundavdrag(ffi: np.ndarray, params: TaxParameters) -> np.ndarray:
    ffi_pbb = ffi / params.pbb
    ga = np.empty_like(ffi)
    ga[ffi_pbb <= 0.99] = 0.423
    mask = (ffi_pbb > 0.99) & (ffi_pbb <= 2.72)
    ga[mask] = 0.225 + 0.2 * ffi_pbb[mask]
    mask = (ffi_pbb > 2.72) & (ffi_pbb <= 3.11)
    ga[mask] = 0.770
    mask = (ffi_pbb > 3.11) & (ffi_pbb <= 7.88)
    ga[mask] = 1.081 - 0.1 * ffi_pbb[mask]
    ga[ffi_pbb > 7.88] = 0.293
    return ga * params.pbb


def jobbskatteavdrag(ai: np.ndarray, ga: np.ndarray, params: TaxParameters) -> np.ndarray:
    pbb = params.pbb
    base = np.zeros_like(ai)
    mask = ai <= 0.91 * pbb
    base[mask] = ai[mask] - ga[mask]
    mask = (ai > 0.91 * pbb) & (ai <= 3.24 * pbb)
    base[mask] = 0.91 * pbb + 0.3874 * (ai[mask] - 0.91 * pbb) - ga[mask]
    mask = (ai > 3.24 * pbb) & (ai <= 8.08 * pbb)
    base[mask] = 1.813 * pbb + 0.199 * (ai[mask] - 3.24 * pbb) - ga[mask]
    mask = ai > 8.08 * pbb
    base[mask] = 2.776 * pbb - ga[mask]
    base = np.maximum(base, 0.0)
    return params.municipal_rate * base


@dataclass(slots=True)
class TaxProfiles:
    gross_monthly: np.ndarray
    net_monthly: np.ndarray
    total_tax_annual: np.ndarray
    average_rate: np.ndarray
    marginal_rate: np.ndarray
    public_pension: np.ndarray
    btp1_low: np.ndarray
    btp1_high: np.ndarray
    btp1_total: np.ndarray
    params: TaxParameters

    def monthly_breakpoints(self) -> dict[str, float]:
        return {
            "PGI cap": self.params.pgi_ceiling / self.params.months_per_year,
            "BTP1 7.5 IBB": self.params.btp1_low_ceiling / self.params.months_per_year,
            "BTP1 30 IBB": self.params.btp1_high_ceiling / self.params.months_per_year,
            "State tax": self.params.brytpunkt / self.params.months_per_year,
            "Fee cap": self.params.fee_cap / self.params.months_per_year,
        }


def compute_profiles(
    gross_monthly: np.ndarray,
    *,
    params: TaxParameters = TaxParameters(),
) -> TaxProfiles:
    months = params.months_per_year
    gross_annual = gross_monthly * months
    ffi = gross_annual
    ga = grundavdrag(ffi, params)
    taxable = np.maximum(ffi - ga, 0.0)

    municipal_tax = params.municipal_rate * taxable
    jsa = jobbskatteavdrag(ffi, ga, params)
    jsa_capped = np.minimum(jsa, municipal_tax)
    state_tax = params.state_rate * np.maximum(taxable - params.skiktgrans, 0.0)
    burial_fee = params.burial_fee * taxable
    public_service = np.minimum(params.public_service_rate * taxable, params.public_service_max)

    total_tax_annual = (municipal_tax - jsa_capped) + state_tax + burial_fee + public_service
    net_annual = gross_annual - total_tax_annual
    net_monthly = net_annual / months

    average_rate = np.divide(
        total_tax_annual, gross_annual, out=np.zeros_like(total_tax_annual), where=gross_annual > 0
    )
    marginal_rate = np.clip(np.gradient(total_tax_annual, gross_annual, edge_order=2), 0.0, 1.0)

    pgi_monthly = np.minimum(params.pgi_rate * gross_monthly, params.pgi_ceiling / months)
    public_pension = params.public_pension_rate * pgi_monthly

    btp1_low = params.btp1_rate_low * np.minimum(gross_monthly, params.btp1_low_ceiling / months)
    btp1_high = params.btp1_rate_high * np.clip(
        gross_monthly - params.btp1_low_ceiling / months,
        0.0,
        params.btp1_high_ceiling / months - params.btp1_low_ceiling / months,
    )
    btp1_total = btp1_low + btp1_high

    return TaxProfiles(
        gross_monthly=gross_monthly,
        net_monthly=net_monthly,
        total_tax_annual=total_tax_annual,
        average_rate=average_rate,
        marginal_rate=marginal_rate,
        public_pension=public_pension,
        btp1_low=btp1_low,
        btp1_high=btp1_high,
        btp1_total=btp1_total,
        params=params,
    )

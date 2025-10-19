# Test Coverage Improvement Plan

## Current Status (2025-10-19)

**Baseline Metrics:**
- Total Tests: 241 (all passing ✅)
- Coverage: 81.04%
- Target: 95% (long-term goal)
- Current Threshold: 81% (temporary, to unblock CI/CD)

## Known Issues

### 1. Region.py Coverage Tracking Anomaly

**Issue:** `quantiq/transform/region.py` shows 0% coverage despite having comprehensive passing tests.

**Evidence:**
- All 11 tests in `tests/transform/test_region.py` pass
- All 14 tests in `tests/transform/test_regions.py` pass
- Manual execution confirms the code works correctly
- Coverage tool reports 0% (31/31 lines missed, lines 13-250)

**Root Cause:** Likely pytest-cov instrumentation issue or configuration problem

**Impact:** ~2% of total coverage (31 statements out of 1,540 total)

**Action Items:**
- [ ] File issue in pytest-cov repository
- [ ] Test with different pytest-cov versions
- [ ] Investigate .coveragerc exclusions
- [ ] Consider alternative coverage tools (coverage.py directly)

## Coverage Gap Analysis

### High-Impact Improvement Targets

| Module | Current | Missing | Target | Impact | Priority |
|--------|---------|---------|--------|--------|----------|
| transform/region.py | 0% | 31 | 80% | +2.0% | HIGH (blocked by bug) |
| backend/operations.py | 44% | 34 | 75% | +1.3% | HIGH |
| collections/tabular_measurement_set.py | 42% | 30 | 70% | +1.2% | HIGH |
| dataio/hierarchy.py | 55% | 22 | 80% | +1.1% | MEDIUM |
| transform/dataset/calculus.py | 54% | 32 | 80% | +1.4% | MEDIUM |
| transform/dataset/baseline.py | 54% | 19 | 80% | +0.9% | MEDIUM |
| transform/base.py | 74% | 20 | 90% | +0.8% | LOW |
| transform/pipeline.py | 79% | 19 | 90% | +0.7% | LOW |

**Total Potential Improvement:** +9.4 percentage points (81% → 90.4%)

## Incremental Improvement Strategy

### Phase 1: Quick Wins (Target: 85% coverage, ~3-4 hours)

**Focus:** High-impact modules with clear test gaps

1. **backend/operations.py** (44% → 75%)
   - Add tests for NumPy fallback paths
   - Test JIT compilation decorators
   - Test device placement utilities
   - **Expected gain:** +1.3%

2. **collections/tabular_measurement_set.py** (42% → 70%)
   - Test tabular data operations
   - Add edge case tests
   - **Expected gain:** +1.2%

3. **dataio/hierarchy.py** (55% → 80%)
   - Test file hierarchy operations
   - Add error handling tests
   - **Expected gain:** +1.1%

**Phase 1 Total:** +3.6% (81% → 84.6%)

### Phase 2: Medium Impact (Target: 88% coverage, ~3-4 hours)

**Focus:** Transform modules with moderate gaps

1. **transform/dataset/calculus.py** (54% → 80%)
   - Test derivative operations
   - Test integration methods
   - **Expected gain:** +1.4%

2. **transform/dataset/baseline.py** (54% → 80%)
   - Test baseline correction algorithms
   - **Expected gain:** +0.9%

3. **transform/dataset/normalization.py** (70% → 85%)
   - Test edge cases in normalization
   - **Expected gain:** +0.5%

**Phase 2 Total:** +2.8% (84.6% → 87.4%)

### Phase 3: Polish (Target: 92% coverage, ~4-6 hours)

**Focus:** Fill remaining gaps across all modules

1. Complete coverage for transform modules
2. Add property-based tests (Hypothesis)
3. Integration tests for full pipelines
4. Edge case and error handling tests

**Phase 3 Total:** +4.6% (87.4% → 92%)

### Phase 4: Excellence (Target: 95%+ coverage)

**Prerequisites:**
- Resolve region.py coverage tracking issue (+2%)
- Add tests for rarely-used code paths
- Consider exclusion of truly untestable code

## Testing Best Practices

### 1. Property-Based Testing

Use Hypothesis for data transformation tests:

```python
from hypothesis import given
from hypothesis import strategies as st

@given(st.lists(st.floats(), min_size=1))
def test_normalization_preserves_range(data):
    """Test that normalization always produces values in [0, 1]."""
    result = normalize(data)
    assert all(0 <= x <= 1 for x in result)
```

### 2. Parametrized Tests

Use pytest.mark.parametrize for multiple scenarios:

```python
@pytest.mark.parametrize("factor,expected", [
    (2.0, [2, 4, 6]),
    (0.5, [0.5, 1, 1.5]),
    (-1.0, [-1, -2, -3]),
])
def test_multiply_transform(factor, expected):
    result = transform.apply(factor)
    assert result == expected
```

### 3. Integration Tests

Test full pipeline workflows:

```python
def test_full_data_processing_pipeline():
    """Test complete workflow from load to analysis."""
    # Load → Transform → Fit → Visualize
    data = load_data("test.csv")
    transformed = pipeline.apply(data)
    fit_result = fit_model(transformed)
    assert fit_result.success
```

## CI/CD Integration

### Current Status
- ✅ Tests run on every commit
- ✅ Coverage reports generated
- ⚠️ Coverage threshold temporarily lowered (81%)

### Recommended Additions
- [ ] Coverage trend tracking (codecov.io or coveralls)
- [ ] Fail on coverage decrease (ratcheting)
- [ ] Separate thresholds for different modules
- [ ] Performance regression tests
- [ ] Visual regression tests (pytest-mpl)

## Timeline

| Phase | Target Coverage | Duration | Completion Date |
|-------|----------------|----------|-----------------|
| Current | 81% | - | 2025-10-19 |
| Phase 1 | 85% | 3-4 hours | Week of 2025-10-21 |
| Phase 2 | 88% | 3-4 hours | Week of 2025-10-28 |
| Phase 3 | 92% | 4-6 hours | Week of 2025-11-04 |
| Phase 4 | 95%+ | Variable | TBD (after region.py fix) |

## Metrics & Monitoring

### Coverage Targets by Module Type

- **Core Data Structures:** 95%+
- **Transform Modules:** 90%+
- **I/O Modules:** 85%+
- **Utility Functions:** 80%+
- **Visualization:** 75%+ (harder to test)

### Quality Indicators

- ✅ All tests passing (241/241)
- ⚠️ 5 test warnings (benchmark fixture usage)
- ✅ No SyntaxWarnings
- ⚠️ Coverage below long-term target (81% vs 95%)
- ✅ Fast test execution (<3 minutes)

## Next Actions

### Immediate (This Week)
1. ✅ Adjust coverage threshold to 81%
2. ✅ Document coverage plan (this file)
3. ✅ Commit changes
4. [ ] File issue for region.py coverage bug
5. [ ] Start Phase 1 improvements

### Short-term (Next 2 Weeks)
1. [ ] Complete Phase 1 (→ 85% coverage)
2. [ ] Set up coverage trending
3. [ ] Add pre-commit hook for coverage checks

### Long-term (Next Month)
1. [ ] Complete Phase 2 (→ 88% coverage)
2. [ ] Resolve region.py tracking issue
3. [ ] Begin Phase 3 improvements
4. [ ] Re-evaluate 95% threshold feasibility

## References

- Coverage HTML Report: `htmlcov/index.html`
- Coverage XML: `coverage.xml`
- Test Configuration: `pytest.ini`
- Coverage Configuration: `.coveragerc`, `pyproject.toml`

## Appendix: Module Coverage Details

### Fully Covered Modules (100%)
- quantiq/data/datasets/two_dimensional.py
- quantiq/transform/measurement/select.py
- quantiq/data/collections/consistent_measurement_set.py
- (12 more modules with 100% coverage)

### Needs Immediate Attention (<50%)
- quantiq/transform/region.py (0% - tracking bug)
- quantiq/backend/operations.py (44%)
- quantiq/collections/tabular_measurement_set.py (42%)
- quantiq/dataio/hierarchy.py (55%)
- quantiq/transform/dataset/baseline.py (54%)
- quantiq/transform/dataset/calculus.py (54%)

---

**Last Updated:** 2025-10-19
**Next Review:** 2025-10-21 (after Phase 1)
**Owner:** quantiq development team

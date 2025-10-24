# piblin-jax Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) documenting key design decisions in piblin_jax.

## What are ADRs?

Architecture Decision Records capture important architectural decisions along with their context and consequences. They help current and future developers understand *why* certain design choices were made.

## Index

### Core Design Decisions

1. **[ADR-001: Backend Abstraction Layer](ADR-001-backend-abstraction.md)** - JAX/NumPy backend abstraction for performance and portability
   - **Status**: Accepted
   - **Impact**: High - Affects all internal array operations
   - **Key Decision**: Transparent JAX/NumPy fallback via unified `jnp` interface

2. **[ADR-002: Immutable Datasets](ADR-002-immutable-datasets.md)** - Copy-on-transform pattern for functional data processing
   - **Status**: Accepted
   - **Impact**: High - Affects all dataset and transform APIs
   - **Key Decision**: All datasets are frozen dataclasses, transforms return new objects

3. **[ADR-003: Six-Level Transform Hierarchy](ADR-003-transform-hierarchy.md)** - Hierarchical transform system from Dataset to ExperimentSet
   - **Status**: Accepted
   - **Impact**: High - Defines the entire transform architecture
   - **Key Decision**: 6 transform levels with composition pattern

4. **[ADR-004: Two-Dictionary Metadata System](ADR-004-metadata-separation.md)** - Separation of conditions vs details
   - **Status**: Accepted
   - **Impact**: Medium - Affects metadata usage throughout
   - **Key Decision**: Separate `conditions` (experimental) from `details` (administrative)

5. **[ADR-005: NumPy API Boundary](ADR-005-numpy-api-boundary.md)** - NumPy external API with JAX internal implementation
   - **Status**: Accepted
   - **Impact**: High - Defines public API contract
   - **Key Decision**: Accept/return NumPy arrays, use JAX internally

## ADR Process

### When to Write an ADR

Write an ADR when making decisions that:
- Affect multiple modules or the entire codebase
- Have long-term consequences
- Involve significant trade-offs
- May be questioned or reconsidered later
- Require explaining to new team members

### ADR Template

```markdown
# ADR-NNN: Title

## Status

Proposed | Accepted | Deprecated | Superseded by ADR-XXX

## Context

What is the issue that we're seeing that is motivating this decision?

## Decision

What is the change that we're proposing and/or doing?

## Consequences

What becomes easier or more difficult to do because of this change?

## Alternatives Considered

What other approaches did we consider?

## References

- Links to relevant issues, PRs, or discussions
```

## Reading Guide

### For New Contributors

Start with these ADRs to understand core architectural principles:

1. **ADR-001** (Backend Abstraction) - Understand why we use `from piblin_jax.backend import jnp`
2. **ADR-002** (Immutable Datasets) - Learn why datasets are frozen and transforms return new objects
3. **ADR-005** (NumPy API Boundary) - Understand the NumPy/JAX interaction

### For Feature Development

- Adding new transforms → Read **ADR-003** (Transform Hierarchy)
- Working with metadata → Read **ADR-004** (Metadata Separation)
- Performance optimization → Read **ADR-001** (Backend Abstraction)

### For API Design

- Public API changes → Read **ADR-005** (NumPy API Boundary)
- Data structures → Read **ADR-002** (Immutable Datasets)

## Relationship Map

```
ADR-001 (Backend Abstraction)
    ├─→ ADR-002 (Immutable Datasets)  [JAX requires immutability]
    └─→ ADR-005 (NumPy API Boundary)  [Backend enables dual APIs]

ADR-002 (Immutable Datasets)
    └─→ ADR-003 (Transform Hierarchy)  [Immutability enables composition]

ADR-003 (Transform Hierarchy)
    └─→ ADR-004 (Metadata Separation)  [Transforms propagate conditions]

ADR-004 (Metadata Separation)
    └─→ [Future] Typed Metadata Schemas
```

## Status Summary

| ADR | Title | Status | Impact |
|-----|-------|--------|--------|
| 001 | Backend Abstraction | ✅ Accepted | High |
| 002 | Immutable Datasets | ✅ Accepted | High |
| 003 | Transform Hierarchy | ✅ Accepted | High |
| 004 | Metadata Separation | ✅ Accepted | Medium |
| 005 | NumPy API Boundary | ✅ Accepted | High |

## Contributing ADRs

When proposing a new ADR:

1. Copy the template above
2. Number it sequentially (ADR-006, etc.)
3. Fill out all sections with detail
4. Consider alternatives thoroughly
5. Open a PR with the ADR for team discussion
6. Update this README to include the new ADR

## References

- [Michael Nygard's ADR proposal](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub organization](https://adr.github.io/)
- [Sustainable Architectural Design Decisions](https://www.infoq.com/articles/sustainable-architectural-design-decisions/)

---

Last Updated: 2024-10-19

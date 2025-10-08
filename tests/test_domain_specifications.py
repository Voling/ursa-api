"""Tests for domain specifications and filtering."""
from __future__ import annotations

import pytest

from app.domain.specifications import (
    Specification, AndSpecification, OrSpecification, NotSpecification,
    filter_by_specification
)


class MockSpecification:
    """Mock specification for testing."""
    
    def __init__(self, should_satisfy: bool):
        self.should_satisfy = should_satisfy
    
    def is_satisfied_by(self, candidate: dict) -> bool:
        return self.should_satisfy
    
    def and_(self, other: Specification) -> Specification:
        return AndSpecification(self, other)
    
    def or_(self, other: Specification) -> Specification:
        return OrSpecification(self, other)
    
    def not_(self) -> Specification:
        return NotSpecification(self)


class TestSpecification:
    """Test base specification functionality."""

    def test_specification_protocol(self):
        """Test that mock specification satisfies protocol."""
        spec = MockSpecification(True)
        
        # Test is_satisfied_by
        assert spec.is_satisfied_by({"test": "data"}) is True
        
        # Test composition methods
        other_spec = MockSpecification(False)
        
        and_spec = spec.and_(other_spec)
        assert isinstance(and_spec, AndSpecification)
        
        or_spec = spec.or_(other_spec)
        assert isinstance(or_spec, OrSpecification)
        
        not_spec = spec.not_()
        assert isinstance(not_spec, NotSpecification)


class TestAndSpecification:
    """Test AND specification composition."""

    def test_and_specification_both_true(self):
        """Test AND specification when both specs are satisfied."""
        spec1 = MockSpecification(True)
        spec2 = MockSpecification(True)
        and_spec = AndSpecification(spec1, spec2)
        
        candidate = {"test": "data"}
        assert and_spec.is_satisfied_by(candidate) is True

    def test_and_specification_first_false(self):
        """Test AND specification when first spec is not satisfied."""
        spec1 = MockSpecification(False)
        spec2 = MockSpecification(True)
        and_spec = AndSpecification(spec1, spec2)
        
        candidate = {"test": "data"}
        assert and_spec.is_satisfied_by(candidate) is False

    def test_and_specification_second_false(self):
        """Test AND specification when second spec is not satisfied."""
        spec1 = MockSpecification(True)
        spec2 = MockSpecification(False)
        and_spec = AndSpecification(spec1, spec2)
        
        candidate = {"test": "data"}
        assert and_spec.is_satisfied_by(candidate) is False

    def test_and_specification_both_false(self):
        """Test AND specification when both specs are not satisfied."""
        spec1 = MockSpecification(False)
        spec2 = MockSpecification(False)
        and_spec = AndSpecification(spec1, spec2)
        
        candidate = {"test": "data"}
        assert and_spec.is_satisfied_by(candidate) is False


class TestOrSpecification:
    """Test OR specification composition."""

    def test_or_specification_both_true(self):
        """Test OR specification when both specs are satisfied."""
        spec1 = MockSpecification(True)
        spec2 = MockSpecification(True)
        or_spec = OrSpecification(spec1, spec2)
        
        candidate = {"test": "data"}
        assert or_spec.is_satisfied_by(candidate) is True

    def test_or_specification_first_true(self):
        """Test OR specification when first spec is satisfied."""
        spec1 = MockSpecification(True)
        spec2 = MockSpecification(False)
        or_spec = OrSpecification(spec1, spec2)
        
        candidate = {"test": "data"}
        assert or_spec.is_satisfied_by(candidate) is True

    def test_or_specification_second_true(self):
        """Test OR specification when second spec is satisfied."""
        spec1 = MockSpecification(False)
        spec2 = MockSpecification(True)
        or_spec = OrSpecification(spec1, spec2)
        
        candidate = {"test": "data"}
        assert or_spec.is_satisfied_by(candidate) is True

    def test_or_specification_both_false(self):
        """Test OR specification when both specs are not satisfied."""
        spec1 = MockSpecification(False)
        spec2 = MockSpecification(False)
        or_spec = OrSpecification(spec1, spec2)
        
        candidate = {"test": "data"}
        assert or_spec.is_satisfied_by(candidate) is False


class TestNotSpecification:
    """Test NOT specification composition."""

    def test_not_specification_true_becomes_false(self):
        """Test NOT specification negates true to false."""
        spec = MockSpecification(True)
        not_spec = NotSpecification(spec)
        
        candidate = {"test": "data"}
        assert not_spec.is_satisfied_by(candidate) is False

    def test_not_specification_false_becomes_true(self):
        """Test NOT specification negates false to true."""
        spec = MockSpecification(False)
        not_spec = NotSpecification(spec)
        
        candidate = {"test": "data"}
        assert not_spec.is_satisfied_by(candidate) is True


class TestComplexSpecifications:
    """Test complex specification compositions."""

    def test_nested_and_or_specifications(self):
        """Test nested AND/OR specifications."""
        spec1 = MockSpecification(True)
        spec2 = MockSpecification(False)
        spec3 = MockSpecification(True)
        
        # (spec1 AND spec2) OR spec3
        and_spec = AndSpecification(spec1, spec2)  # True AND False = False
        or_spec = OrSpecification(and_spec, spec3)  # False OR True = True
        
        candidate = {"test": "data"}
        assert or_spec.is_satisfied_by(candidate) is True

    def test_not_with_and_specifications(self):
        """Test NOT with AND specifications."""
        spec1 = MockSpecification(True)
        spec2 = MockSpecification(True)
        
        # NOT (spec1 AND spec2)
        and_spec = AndSpecification(spec1, spec2)  # True AND True = True
        not_spec = NotSpecification(and_spec)  # NOT True = False
        
        candidate = {"test": "data"}
        assert not_spec.is_satisfied_by(candidate) is False

    def test_chain_method_calls(self):
        """Test chaining specification method calls."""
        spec1 = MockSpecification(True)
        spec2 = MockSpecification(False)
        spec3 = MockSpecification(True)
        
        # spec1.and_(spec2).or_(spec3)
        result = spec1.and_(spec2).or_(spec3)
        
        candidate = {"test": "data"}
        # (True AND False) OR True = False OR True = True
        assert result.is_satisfied_by(candidate) is True


class TestFilterBySpecification:
    """Test the filter_by_specification helper function."""

    def test_filter_empty_list(self):
        """Test filtering empty list."""
        spec = MockSpecification(True)
        items = []
        
        result = filter_by_specification(items, spec)
        assert result == []

    def test_filter_all_satisfy(self):
        """Test filtering when all items satisfy specification."""
        spec = MockSpecification(True)
        items = [
            {"id": "item1", "name": "Item 1"},
            {"id": "item2", "name": "Item 2"},
            {"id": "item3", "name": "Item 3"},
        ]
        
        result = filter_by_specification(items, spec)
        assert result == items

    def test_filter_none_satisfy(self):
        """Test filtering when no items satisfy specification."""
        spec = MockSpecification(False)
        items = [
            {"id": "item1", "name": "Item 1"},
            {"id": "item2", "name": "Item 2"},
            {"id": "item3", "name": "Item 3"},
        ]
        
        result = filter_by_specification(items, spec)
        assert result == []

    def test_filter_mixed_results(self):
        """Test filtering with mixed results (some satisfy, some don't)."""
        # Create a spec that only satisfies items with "Item 1" or "Item 3"
        class NameSpecification:
            def is_satisfied_by(self, candidate: dict) -> bool:
                return candidate.get("name") in ["Item 1", "Item 3"]
            
            def and_(self, other):
                return AndSpecification(self, other)
            
            def or_(self, other):
                return OrSpecification(self, other)
            
            def not_(self):
                return NotSpecification(self)
        
        spec = NameSpecification()
        items = [
            {"id": "item1", "name": "Item 1"},
            {"id": "item2", "name": "Item 2"},
            {"id": "item3", "name": "Item 3"},
            {"id": "item4", "name": "Item 4"},
        ]
        
        result = filter_by_specification(items, spec)
        assert len(result) == 2
        assert result[0]["name"] == "Item 1"
        assert result[1]["name"] == "Item 3"

    def test_filter_with_complex_specification(self):
        """Test filtering with complex composed specification."""
        # Create specs for different criteria
        class IdSpecification:
            def __init__(self, allowed_ids):
                self.allowed_ids = allowed_ids
            
            def is_satisfied_by(self, candidate: dict) -> bool:
                return candidate.get("id") in self.allowed_ids
            
            def and_(self, other):
                return AndSpecification(self, other)
            
            def or_(self, other):
                return OrSpecification(self, other)
            
            def not_(self):
                return NotSpecification(self)
        
        class StatusSpecification:
            def __init__(self, allowed_status):
                self.allowed_status = allowed_status
            
            def is_satisfied_by(self, candidate: dict) -> bool:
                return candidate.get("status") == self.allowed_status
            
            def and_(self, other):
                return AndSpecification(self, other)
            
            def or_(self, other):
                return OrSpecification(self, other)
            
            def not_(self):
                return NotSpecification(self)
        
        items = [
            {"id": "item1", "name": "Item 1", "status": "active"},
            {"id": "item2", "name": "Item 2", "status": "inactive"},
            {"id": "item3", "name": "Item 3", "status": "active"},
            {"id": "item4", "name": "Item 4", "status": "pending"},
        ]
        
        # Filter for items with id in ["item1", "item3"] AND status == "active"
        id_spec = IdSpecification(["item1", "item3"])
        status_spec = StatusSpecification("active")
        combined_spec = id_spec.and_(status_spec)
        
        result = filter_by_specification(items, combined_spec)
        assert len(result) == 2
        assert result[0]["id"] == "item1"
        assert result[1]["id"] == "item3"

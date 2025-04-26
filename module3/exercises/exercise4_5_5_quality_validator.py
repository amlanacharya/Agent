"""
Exercise 4.5.5: Quality Validator

This exercise implements a response quality validator that checks for clarity,
conciseness, and helpfulness in agent responses.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
import re
import math
from datetime import datetime


class QualityDimension(str, Enum):
    """Dimensions of response quality that can be evaluated."""
    CLARITY = "clarity"           # How clear and understandable the response is
    CONCISENESS = "conciseness"   # How concise and to-the-point the response is
    HELPFULNESS = "helpfulness"   # How helpful and useful the response is
    COHERENCE = "coherence"       # How logically organized and connected the response is
    ENGAGEMENT = "engagement"     # How engaging and appropriate the tone is


class QualityLevel(str, Enum):
    """Quality levels for evaluating responses."""
    EXCELLENT = "excellent"  # Exceptional quality
    GOOD = "good"           # Good quality with minor issues
    ADEQUATE = "adequate"   # Acceptable quality with some issues
    POOR = "poor"           # Poor quality with significant issues
    UNACCEPTABLE = "unacceptable"  # Unacceptable quality with major issues


class QualityIssue(BaseModel):
    """A specific issue identified in the response quality."""
    dimension: QualityDimension = Field(..., description="The quality dimension affected")
    description: str = Field(..., description="Description of the issue")
    severity: float = Field(..., ge=0.0, le=1.0, description="Severity of the issue (0-1)")
    suggestion: Optional[str] = Field(None, description="Suggestion for improvement")
    location: Optional[Tuple[int, int]] = Field(None, description="Start and end indices of the issue in the text")


class QualityMetrics(BaseModel):
    """Metrics for evaluating the quality of a response."""
    clarity_score: float = Field(..., ge=0.0, le=1.0, description="Score for clarity (0-1)")
    conciseness_score: float = Field(..., ge=0.0, le=1.0, description="Score for conciseness (0-1)")
    helpfulness_score: float = Field(..., ge=0.0, le=1.0, description="Score for helpfulness (0-1)")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Score for coherence (0-1)")
    engagement_score: float = Field(..., ge=0.0, le=1.0, description="Score for engagement (0-1)")

    overall_quality_level: QualityLevel = Field(..., description="Overall quality level")
    issues: List[QualityIssue] = Field(default_factory=list, description="List of identified quality issues")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the quality was evaluated")

    @property
    def overall_score(self) -> float:
        """Calculate the overall quality score as a weighted average of dimension scores."""
        # Define weights for each dimension (can be adjusted based on importance)
        weights = {
            "clarity": 0.25,
            "helpfulness": 0.25,
            "conciseness": 0.2,
            "coherence": 0.15,
            "engagement": 0.15
        }

        # Calculate weighted score
        weighted_score = (
            self.clarity_score * weights["clarity"] +
            self.helpfulness_score * weights["helpfulness"] +
            self.conciseness_score * weights["conciseness"] +
            self.coherence_score * weights["coherence"] +
            self.engagement_score * weights["engagement"]
        )

        return weighted_score

    @model_validator(mode='after')
    def determine_quality_level(self):
        """Determine the overall quality level based on scores and issues."""
        score = self.overall_score

        # Determine quality level based on overall score
        if score >= 0.9:
            self.overall_quality_level = QualityLevel.EXCELLENT
        elif score >= 0.75:
            self.overall_quality_level = QualityLevel.GOOD
        elif score >= 0.6:
            self.overall_quality_level = QualityLevel.ADEQUATE
        elif score >= 0.4:
            self.overall_quality_level = QualityLevel.POOR
        else:
            self.overall_quality_level = QualityLevel.UNACCEPTABLE

        # Adjust for severe issues
        severe_issues = [issue for issue in self.issues if issue.severity > 0.7]
        if severe_issues:
            # Downgrade by one level if there are severe issues
            if self.overall_quality_level == QualityLevel.EXCELLENT:
                self.overall_quality_level = QualityLevel.GOOD
            elif self.overall_quality_level == QualityLevel.GOOD:
                self.overall_quality_level = QualityLevel.ADEQUATE
            elif self.overall_quality_level == QualityLevel.ADEQUATE:
                self.overall_quality_level = QualityLevel.POOR

        return self

    def get_dimension_score(self, dimension: QualityDimension) -> float:
        """Get the score for a specific quality dimension."""
        if dimension == QualityDimension.CLARITY:
            return self.clarity_score
        elif dimension == QualityDimension.CONCISENESS:
            return self.conciseness_score
        elif dimension == QualityDimension.HELPFULNESS:
            return self.helpfulness_score
        elif dimension == QualityDimension.COHERENCE:
            return self.coherence_score
        elif dimension == QualityDimension.ENGAGEMENT:
            return self.engagement_score
        else:
            raise ValueError(f"Unknown dimension: {dimension}")

    def get_issues_by_dimension(self, dimension: QualityDimension) -> List[QualityIssue]:
        """Get issues related to a specific quality dimension."""
        return [issue for issue in self.issues if issue.dimension == dimension]

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the quality metrics."""
        return {
            "overall_score": self.overall_score,
            "overall_quality_level": self.overall_quality_level,
            "dimension_scores": {
                "clarity": self.clarity_score,
                "conciseness": self.conciseness_score,
                "helpfulness": self.helpfulness_score,
                "coherence": self.coherence_score,
                "engagement": self.engagement_score
            },
            "issue_count": len(self.issues),
            "severe_issue_count": len([issue for issue in self.issues if issue.severity > 0.7])
        }

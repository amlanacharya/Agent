"""
Exercise 4.5.2: Content Moderation System

This exercise implements a system that checks agent outputs for inappropriate content,
bias, and factual accuracy.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import re
from datetime import datetime


class ContentCategory(str, Enum):
    """Categories of content to moderate."""
    PROFANITY = "profanity"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    PERSONAL_INFO = "personal_info"
    MISINFORMATION = "misinformation"
    BIAS = "bias"
    POLITICAL = "political"
    RELIGIOUS = "religious"
    SPAM = "spam"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"


class SeverityLevel(str, Enum):
    """Severity levels for content moderation issues."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentFlag(BaseModel):
    """Model for a content moderation flag."""
    category: ContentCategory = Field(..., description="Category of the flagged content")
    severity: SeverityLevel = Field(..., description="Severity level of the issue")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the flag")
    snippet: Optional[str] = Field(None, description="Snippet of the flagged content")
    start_pos: Optional[int] = Field(None, description="Start position of the flagged content")
    end_pos: Optional[int] = Field(None, description="End position of the flagged content")
    details: Optional[str] = Field(None, description="Additional details about the flag")


class BiasType(str, Enum):
    """Types of bias to detect."""
    GENDER = "gender"
    RACIAL = "racial"
    RELIGIOUS = "religious"
    POLITICAL = "political"
    AGE = "age"
    SOCIOECONOMIC = "socioeconomic"
    ABILITY = "ability"
    CULTURAL = "cultural"
    GEOGRAPHICAL = "geographical"
    EDUCATIONAL = "educational"


class BiasFlag(BaseModel):
    """Model for a bias flag."""
    bias_type: BiasType = Field(..., description="Type of bias detected")
    severity: SeverityLevel = Field(..., description="Severity level of the bias")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the flag")
    snippet: Optional[str] = Field(None, description="Snippet of the biased content")
    start_pos: Optional[int] = Field(None, description="Start position of the biased content")
    end_pos: Optional[int] = Field(None, description="End position of the biased content")
    details: Optional[str] = Field(None, description="Additional details about the bias")


class FactualErrorType(str, Enum):
    """Types of factual errors to detect."""
    INCORRECT_FACT = "incorrect_fact"
    OUTDATED_INFORMATION = "outdated_information"
    MISLEADING_STATEMENT = "misleading_statement"
    UNVERIFIED_CLAIM = "unverified_claim"
    LOGICAL_FALLACY = "logical_fallacy"
    STATISTICAL_ERROR = "statistical_error"
    MISATTRIBUTION = "misattribution"
    CONTRADICTION = "contradiction"


class FactualErrorFlag(BaseModel):
    """Model for a factual error flag."""
    error_type: FactualErrorType = Field(..., description="Type of factual error detected")
    severity: SeverityLevel = Field(..., description="Severity level of the error")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the flag")
    snippet: Optional[str] = Field(None, description="Snippet of the erroneous content")
    start_pos: Optional[int] = Field(None, description="Start position of the erroneous content")
    end_pos: Optional[int] = Field(None, description="End position of the erroneous content")
    details: Optional[str] = Field(None, description="Additional details about the error")
    correction: Optional[str] = Field(None, description="Suggested correction")


class ContentModerationResult(BaseModel):
    """Model for content moderation results."""
    text: str = Field(..., description="Original text that was moderated")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the moderation")
    content_flags: List[ContentFlag] = Field(default_factory=list, description="Content moderation flags")
    bias_flags: List[BiasFlag] = Field(default_factory=list, description="Bias flags")
    factual_error_flags: List[FactualErrorFlag] = Field(default_factory=list, description="Factual error flags")
    overall_severity: SeverityLevel = Field(default=SeverityLevel.NONE, description="Overall severity level")
    is_approved: bool = Field(..., description="Whether the content is approved for release")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection if not approved")
    
    @model_validator(mode='after')
    def calculate_overall_severity(self):
        """Calculate the overall severity level based on all flags."""
        all_flags = self.content_flags + self.bias_flags + self.factual_error_flags
        
        if not all_flags:
            self.overall_severity = SeverityLevel.NONE
            return self
        
        # Map severity levels to numeric values
        severity_values = {
            SeverityLevel.NONE: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4
        }
        
        # Get the highest severity level
        max_severity = max(severity_values[flag.severity] for flag in all_flags)
        
        # Map back to SeverityLevel
        for level, value in severity_values.items():
            if value == max_severity:
                self.overall_severity = level
                break
        
        return self
    
    @property
    def has_flags(self) -> bool:
        """Check if there are any flags."""
        return bool(self.content_flags or self.bias_flags or self.factual_error_flags)
    
    @property
    def total_flags(self) -> int:
        """Get the total number of flags."""
        return len(self.content_flags) + len(self.bias_flags) + len(self.factual_error_flags)
    
    @property
    def categories_flagged(self) -> Set[ContentCategory]:
        """Get the set of flagged content categories."""
        return {flag.category for flag in self.content_flags}
    
    @property
    def bias_types_flagged(self) -> Set[BiasType]:
        """Get the set of flagged bias types."""
        return {flag.bias_type for flag in self.bias_flags}
    
    @property
    def error_types_flagged(self) -> Set[FactualErrorType]:
        """Get the set of flagged factual error types."""
        return {flag.error_type for flag in self.factual_error_flags}
    
    def get_flags_by_severity(self, severity: SeverityLevel) -> List[Any]:
        """
        Get all flags with a specific severity level.
        
        Args:
            severity: Severity level to filter by
            
        Returns:
            List of flags with the specified severity
        """
        return [
            flag for flag in self.content_flags + self.bias_flags + self.factual_error_flags
            if flag.severity == severity
        ]
    
    def get_content_flags_by_category(self, category: ContentCategory) -> List[ContentFlag]:
        """
        Get content flags for a specific category.
        
        Args:
            category: Content category to filter by
            
        Returns:
            List of content flags for the specified category
        """
        return [flag for flag in self.content_flags if flag.category == category]
    
    def get_bias_flags_by_type(self, bias_type: BiasType) -> List[BiasFlag]:
        """
        Get bias flags for a specific type.
        
        Args:
            bias_type: Bias type to filter by
            
        Returns:
            List of bias flags for the specified type
        """
        return [flag for flag in self.bias_flags if flag.bias_type == bias_type]
    
    def get_factual_error_flags_by_type(self, error_type: FactualErrorType) -> List[FactualErrorFlag]:
        """
        Get factual error flags for a specific type.
        
        Args:
            error_type: Factual error type to filter by
            
        Returns:
            List of factual error flags for the specified type
        """
        return [flag for flag in self.factual_error_flags if flag.error_type == error_type]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the moderation results.
        
        Returns:
            Dictionary with summary information
        """
        return {
            "total_flags": self.total_flags,
            "content_flags": len(self.content_flags),
            "bias_flags": len(self.bias_flags),
            "factual_error_flags": len(self.factual_error_flags),
            "overall_severity": self.overall_severity,
            "is_approved": self.is_approved,
            "categories_flagged": list(self.categories_flagged),
            "bias_types_flagged": list(self.bias_types_flagged),
            "error_types_flagged": list(self.error_types_flagged)
        }


class ContentModerationSystem(BaseModel):
    """
    System for moderating agent outputs.
    
    This system checks content for inappropriate material, bias, and factual errors.
    """
    profanity_list: List[str] = Field(default_factory=list, description="List of profane words to detect")
    hate_speech_patterns: List[str] = Field(default_factory=list, description="Patterns for detecting hate speech")
    violence_patterns: List[str] = Field(default_factory=list, description="Patterns for detecting violent content")
    sexual_patterns: List[str] = Field(default_factory=list, description="Patterns for detecting sexual content")
    personal_info_patterns: List[str] = Field(default_factory=list, description="Patterns for detecting personal info")
    bias_patterns: Dict[BiasType, List[str]] = Field(default_factory=dict, description="Patterns for detecting bias")
    factual_errors_db: Dict[str, str] = Field(default_factory=dict, description="Database of known factual errors")
    approval_threshold: SeverityLevel = Field(default=SeverityLevel.MEDIUM, description="Threshold for content approval")
    
    def __init__(self, **data):
        """Initialize with default patterns if not provided."""
        super().__init__(**data)
        
        # Initialize default patterns if not provided
        if not self.profanity_list:
            self.profanity_list = ["profanity1", "profanity2", "profanity3"]  # Placeholder
        
        if not self.hate_speech_patterns:
            self.hate_speech_patterns = [
                r'\b(hate|despise)\s+(\w+\s+)?(race|religion|gender|orientation)\b',
                r'\b(all|every)\s+(\w+\s+)?(people|person)s?\s+are\s+(\w+\s+)?(bad|evil|stupid|dumb)\b'
            ]
        
        if not self.violence_patterns:
            self.violence_patterns = [
                r'\b(kill|murder|hurt|harm|injure|attack)\s+(\w+\s+)?(people|person)s?\b',
                r'\b(bomb|shoot|stab|assault)\b'
            ]
        
        if not self.sexual_patterns:
            self.sexual_patterns = [
                r'\b(explicit sexual terms)\b'  # Placeholder
            ]
        
        if not self.personal_info_patterns:
            self.personal_info_patterns = [
                r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # SSN
                r'\b\d{16}\b',  # Credit card
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # Phone number
            ]
        
        if not self.bias_patterns:
            self.bias_patterns = {
                BiasType.GENDER: [
                    r'\b(all|every)\s+(men|women|males|females)\s+are\s+(\w+\s+)?(bad|good|better|worse)\b',
                    r'\b(men|women)\s+can\'?t\s+(\w+\s+)?(do|understand|perform)\b'
                ],
                BiasType.RACIAL: [
                    r'\b(all|every)\s+(\w+\s+)?(race|ethnicity)\s+is\s+(\w+\s+)?(bad|good|better|worse)\b'
                ],
                BiasType.RELIGIOUS: [
                    r'\b(all|every)\s+(\w+\s+)?(religion|religious people)\s+are\s+(\w+\s+)?(bad|good|better|worse)\b'
                ]
            }
        
        if not self.factual_errors_db:
            self.factual_errors_db = {
                "the earth is flat": "The Earth is an oblate spheroid, not flat.",
                "vaccines cause autism": "Scientific consensus is that vaccines do not cause autism.",
                "climate change is a hoax": "Climate change is supported by overwhelming scientific evidence."
            }
    
    def moderate_content(self, text: str) -> ContentModerationResult:
        """
        Moderate content for inappropriate material, bias, and factual errors.
        
        Args:
            text: Text content to moderate
            
        Returns:
            ContentModerationResult with moderation results
        """
        content_flags = []
        bias_flags = []
        factual_error_flags = []
        
        # Check for inappropriate content
        content_flags.extend(self._check_profanity(text))
        content_flags.extend(self._check_hate_speech(text))
        content_flags.extend(self._check_violence(text))
        content_flags.extend(self._check_sexual_content(text))
        content_flags.extend(self._check_personal_info(text))
        
        # Check for bias
        bias_flags.extend(self._check_bias(text))
        
        # Check for factual errors
        factual_error_flags.extend(self._check_factual_errors(text))
        
        # Determine if content is approved
        is_approved = True
        rejection_reason = None
        
        # Map severity levels to numeric values
        severity_values = {
            SeverityLevel.NONE: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4
        }
        
        # Check if any flags exceed the approval threshold
        all_flags = content_flags + bias_flags + factual_error_flags
        if all_flags:
            max_severity = max(severity_values[flag.severity] for flag in all_flags)
            max_severity_level = next(level for level, value in severity_values.items() if value == max_severity)
            
            if severity_values[max_severity_level] >= severity_values[self.approval_threshold]:
                is_approved = False
                rejection_reason = f"Content contains {max_severity_level.value} severity issues"
        
        return ContentModerationResult(
            text=text,
            content_flags=content_flags,
            bias_flags=bias_flags,
            factual_error_flags=factual_error_flags,
            is_approved=is_approved,
            rejection_reason=rejection_reason
        )
    
    def _check_profanity(self, text: str) -> List[ContentFlag]:
        """Check for profanity in text."""
        flags = []
        
        for word in self.profanity_list:
            pattern = r'\b' + re.escape(word) + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                flags.append(ContentFlag(
                    category=ContentCategory.PROFANITY,
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.9,
                    snippet=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    details=f"Profane word detected: {match.group(0)}"
                ))
        
        return flags
    
    def _check_hate_speech(self, text: str) -> List[ContentFlag]:
        """Check for hate speech in text."""
        flags = []
        
        for pattern in self.hate_speech_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                flags.append(ContentFlag(
                    category=ContentCategory.HATE_SPEECH,
                    severity=SeverityLevel.HIGH,
                    confidence=0.8,
                    snippet=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    details=f"Potential hate speech detected: {match.group(0)}"
                ))
        
        return flags
    
    def _check_violence(self, text: str) -> List[ContentFlag]:
        """Check for violent content in text."""
        flags = []
        
        for pattern in self.violence_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                flags.append(ContentFlag(
                    category=ContentCategory.VIOLENCE,
                    severity=SeverityLevel.HIGH,
                    confidence=0.7,
                    snippet=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    details=f"Potential violent content detected: {match.group(0)}"
                ))
        
        return flags
    
    def _check_sexual_content(self, text: str) -> List[ContentFlag]:
        """Check for sexual content in text."""
        flags = []
        
        for pattern in self.sexual_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                flags.append(ContentFlag(
                    category=ContentCategory.SEXUAL,
                    severity=SeverityLevel.HIGH,
                    confidence=0.7,
                    snippet=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    details=f"Potential sexual content detected: {match.group(0)}"
                ))
        
        return flags
    
    def _check_personal_info(self, text: str) -> List[ContentFlag]:
        """Check for personal information in text."""
        flags = []
        
        for pattern in self.personal_info_patterns:
            for match in re.finditer(pattern, text):
                flags.append(ContentFlag(
                    category=ContentCategory.PERSONAL_INFO,
                    severity=SeverityLevel.CRITICAL,
                    confidence=0.9,
                    snippet=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    details=f"Potential personal information detected: {match.group(0)}"
                ))
        
        return flags
    
    def _check_bias(self, text: str) -> List[BiasFlag]:
        """Check for bias in text."""
        flags = []
        
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    flags.append(BiasFlag(
                        bias_type=bias_type,
                        severity=SeverityLevel.MEDIUM,
                        confidence=0.7,
                        snippet=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        details=f"Potential {bias_type.value} bias detected: {match.group(0)}"
                    ))
        
        return flags
    
    def _check_factual_errors(self, text: str) -> List[FactualErrorFlag]:
        """Check for factual errors in text."""
        flags = []
        
        # Check against known factual errors
        for error_text, correction in self.factual_errors_db.items():
            if error_text.lower() in text.lower():
                # Find the position of the error
                start_pos = text.lower().find(error_text.lower())
                end_pos = start_pos + len(error_text)
                
                flags.append(FactualErrorFlag(
                    error_type=FactualErrorType.INCORRECT_FACT,
                    severity=SeverityLevel.HIGH,
                    confidence=0.9,
                    snippet=text[start_pos:end_pos],
                    start_pos=start_pos,
                    end_pos=end_pos,
                    details=f"Factual error detected: {error_text}",
                    correction=correction
                ))
        
        # Check for unverified claims (simplified example)
        unverified_patterns = [
            r'\b(studies show|research proves|scientists confirm)\b',
            r'\b(everyone knows|it is known|obviously)\b'
        ]
        
        for pattern in unverified_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                flags.append(FactualErrorFlag(
                    error_type=FactualErrorType.UNVERIFIED_CLAIM,
                    severity=SeverityLevel.LOW,
                    confidence=0.6,
                    snippet=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    details=f"Potential unverified claim: {match.group(0)}",
                    correction="Consider adding specific sources or evidence for this claim."
                ))
        
        return flags


# Example usage
if __name__ == "__main__":
    # Create a content moderation system
    moderation_system = ContentModerationSystem()
    
    # Example 1: Clean content
    clean_text = "The weather in New York is currently sunny with a temperature of 75°F."
    clean_result = moderation_system.moderate_content(clean_text)
    
    print("Example 1: Clean content")
    print(f"Text: '{clean_text}'")
    print(f"Is approved: {clean_result.is_approved}")
    print(f"Total flags: {clean_result.total_flags}")
    print(f"Overall severity: {clean_result.overall_severity}")
    print()
    
    # Example 2: Content with profanity
    profanity_text = "The weather is profanity1 hot today."
    profanity_result = moderation_system.moderate_content(profanity_text)
    
    print("Example 2: Content with profanity")
    print(f"Text: '{profanity_text}'")
    print(f"Is approved: {profanity_result.is_approved}")
    print(f"Total flags: {profanity_result.total_flags}")
    print(f"Overall severity: {profanity_result.overall_severity}")
    if profanity_result.content_flags:
        print("Content flags:")
        for flag in profanity_result.content_flags:
            print(f"  - Category: {flag.category}")
            print(f"    Severity: {flag.severity}")
            print(f"    Snippet: '{flag.snippet}'")
            print(f"    Details: {flag.details}")
    print()
    
    # Example 3: Content with factual error
    factual_error_text = "The earth is flat and climate change is a hoax."
    factual_error_result = moderation_system.moderate_content(factual_error_text)
    
    print("Example 3: Content with factual error")
    print(f"Text: '{factual_error_text}'")
    print(f"Is approved: {factual_error_result.is_approved}")
    print(f"Total flags: {factual_error_result.total_flags}")
    print(f"Overall severity: {factual_error_result.overall_severity}")
    if factual_error_result.factual_error_flags:
        print("Factual error flags:")
        for flag in factual_error_result.factual_error_flags:
            print(f"  - Error type: {flag.error_type}")
            print(f"    Severity: {flag.severity}")
            print(f"    Snippet: '{flag.snippet}'")
            print(f"    Details: {flag.details}")
            print(f"    Correction: {flag.correction}")
    print()
    
    # Example 4: Content with bias
    bias_text = "All women can't understand complex technical concepts."
    bias_result = moderation_system.moderate_content(bias_text)
    
    print("Example 4: Content with bias")
    print(f"Text: '{bias_text}'")
    print(f"Is approved: {bias_result.is_approved}")
    print(f"Total flags: {bias_result.total_flags}")
    print(f"Overall severity: {bias_result.overall_severity}")
    if bias_result.bias_flags:
        print("Bias flags:")
        for flag in bias_result.bias_flags:
            print(f"  - Bias type: {flag.bias_type}")
            print(f"    Severity: {flag.severity}")
            print(f"    Snippet: '{flag.snippet}'")
            print(f"    Details: {flag.details}")
    print()

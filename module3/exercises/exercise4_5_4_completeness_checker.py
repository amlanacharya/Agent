"""
Exercise 4.5.4: Completeness Checker

This exercise implements a system that verifies all parts of a multi-part question
are addressed in the response.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
import re
from datetime import datetime


class QuestionType(str, Enum):
    """Types of questions that can be asked."""
    FACTUAL = "factual"          # Questions about facts (what, who, where)
    TEMPORAL = "temporal"        # Questions about time (when)
    PROCEDURAL = "procedural"    # Questions about how to do something (how)
    CAUSAL = "causal"            # Questions about causes (why)
    COMPARATIVE = "comparative"  # Questions comparing things
    HYPOTHETICAL = "hypothetical"  # Questions about hypothetical scenarios
    PREFERENCE = "preference"    # Questions about preferences
    CLARIFICATION = "clarification"  # Questions seeking clarification
    CONFIRMATION = "confirmation"  # Questions seeking confirmation (yes/no)
    QUANTITATIVE = "quantitative"  # Questions about quantities (how many, how much)


class Question(BaseModel):
    """A question extracted from user input."""
    text: str = Field(..., description="The full text of the question")
    question_type: QuestionType = Field(..., description="Type of question")
    key_entities: List[str] = Field(default_factory=list, description="Key entities mentioned in the question")
    key_topics: List[str] = Field(default_factory=list, description="Key topics of the question")
    is_multi_part: bool = Field(default=False, description="Whether this is a multi-part question")
    sub_questions: List["Question"] = Field(default_factory=list, description="Sub-questions for multi-part questions")
    
    @property
    def is_addressed(self) -> bool:
        """Check if the question has been addressed."""
        # This will be set by the CompletenessChecker
        return getattr(self, "_is_addressed", False)
    
    @is_addressed.setter
    def is_addressed(self, value: bool):
        """Set whether the question has been addressed."""
        self._is_addressed = value


class UserQuery(BaseModel):
    """A query from a user, which may contain multiple questions."""
    text: str = Field(..., description="The full text of the user query")
    questions: List[Question] = Field(default_factory=list, description="Questions extracted from the query")
    normalized_text: str = Field("", description="Normalized version of the query text")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the query was received")
    
    @model_validator(mode='after')
    def extract_questions(self):
        """Extract questions from the query text."""
        # Normalize text
        self.normalized_text = self._normalize_text(self.text)
        
        # Extract explicit questions (with question marks)
        explicit_questions = self._extract_explicit_questions(self.text)
        
        # If no explicit questions, check for implicit questions
        if not explicit_questions:
            implicit_questions = self._extract_implicit_questions(self.text)
            self.questions.extend(implicit_questions)
        else:
            self.questions.extend(explicit_questions)
        
        # Extract key entities and topics for each question
        for question in self.questions:
            question.key_entities = self._extract_entities(question.text)
            question.key_topics = self._extract_topics(question.text)
        
        return self
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and converting to lowercase."""
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _extract_explicit_questions(self, text: str) -> List[Question]:
        """Extract explicit questions (with question marks) from text."""
        questions = []
        
        # Find all question segments
        question_pattern = r'([^.!?]+\?)'
        matches = re.findall(question_pattern, text)
        
        for match in matches:
            question_text = match.strip()
            question_type = self._determine_question_type(question_text)
            
            # Check if this is a multi-part question
            sub_questions = self._extract_sub_questions(question_text)
            is_multi_part = len(sub_questions) > 0
            
            question = Question(
                text=question_text,
                question_type=question_type,
                is_multi_part=is_multi_part,
                sub_questions=sub_questions
            )
            
            questions.append(question)
        
        return questions
    
    def _extract_implicit_questions(self, text: str) -> List[Question]:
        """Extract implicit questions (without question marks) from text."""
        questions = []
        
        # Check for question words
        question_words = ["who", "what", "when", "where", "why", "how", "can", "could", "would", "should", "is", "are", "do", "does"]
        
        # If text starts with a question word, treat the whole text as a question
        words = text.lower().split()
        if words and words[0] in question_words:
            question_type = self._determine_question_type(text)
            
            question = Question(
                text=text,
                question_type=question_type
            )
            
            questions.append(question)
        
        return questions
    
    def _extract_sub_questions(self, text: str) -> List[Question]:
        """Extract sub-questions from a multi-part question."""
        sub_questions = []
        
        # Look for conjunctions followed by question words
        sub_question_pattern = r'(?:and|or|also|plus|additionally)\s+((?:who|what|when|where|why|how|can|could|would|should|is|are|do|does)[^?,.;]+)'
        matches = re.finditer(sub_question_pattern, text, re.IGNORECASE)
        
        for match in matches:
            sub_question_text = match.group(1).strip()
            question_type = self._determine_question_type(sub_question_text)
            
            sub_question = Question(
                text=sub_question_text,
                question_type=question_type
            )
            
            sub_questions.append(sub_question)
        
        return sub_questions
    
    def _determine_question_type(self, text: str) -> QuestionType:
        """Determine the type of a question based on its text."""
        text_lower = text.lower()
        
        # Check for question types based on question words and patterns
        if re.search(r'\b(what|who|which|where)\b', text_lower):
            return QuestionType.FACTUAL
        elif re.search(r'\b(when)\b', text_lower):
            return QuestionType.TEMPORAL
        elif re.search(r'\b(how to|how do|how can|how should|how would)\b', text_lower):
            return QuestionType.PROCEDURAL
        elif re.search(r'\b(why|how come)\b', text_lower):
            return QuestionType.CAUSAL
        elif re.search(r'\b(compare|difference|better|worse|prefer)\b', text_lower):
            return QuestionType.COMPARATIVE
        elif re.search(r'\b(if|would|imagine|suppose)\b', text_lower):
            return QuestionType.HYPOTHETICAL
        elif re.search(r'\b(do you think|recommend|suggest|prefer|like|favorite)\b', text_lower):
            return QuestionType.PREFERENCE
        elif re.search(r'\b(mean|clarify|explain|elaborate)\b', text_lower):
            return QuestionType.CLARIFICATION
        elif re.search(r'\b(is it|are they|do you|can you|will|should)\b', text_lower) and len(text_lower.split()) < 10:
            return QuestionType.CONFIRMATION
        elif re.search(r'\b(how many|how much|count|number|quantity)\b', text_lower):
            return QuestionType.QUANTITATIVE
        
        # Default to factual if no specific type is determined
        return QuestionType.FACTUAL
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text (simplified version)."""
        # Remove question words and stop words
        stop_words = ["who", "what", "when", "where", "why", "how", "is", "are", "the", "a", "an", 
                      "in", "on", "at", "to", "for", "with", "by", "about", "like", "as", "of", "and", "or"]
        
        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Split into words
        words = clean_text.split()
        
        # Filter out stop words and short words
        entities = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return unique entities
        return list(set(entities))
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text (simplified version)."""
        # This is a simplified implementation
        # In a real system, this would use NLP techniques like topic modeling
        
        # Define some common topics and their related words
        topics = {
            "weather": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", "storm", "humidity", "wind"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel", "booking", "destination", "tourism", "visit"],
            "food": ["food", "restaurant", "recipe", "cooking", "meal", "dinner", "lunch", "breakfast", "ingredient"],
            "technology": ["technology", "computer", "software", "hardware", "app", "device", "digital", "internet", "online"],
            "health": ["health", "medical", "doctor", "symptom", "disease", "treatment", "medicine", "wellness", "fitness"],
            "finance": ["finance", "money", "bank", "investment", "stock", "budget", "saving", "expense", "financial"],
            "education": ["education", "school", "university", "college", "course", "study", "learning", "student", "teacher"],
            "entertainment": ["entertainment", "movie", "music", "game", "show", "concert", "theater", "play", "performance"]
        }
        
        # Check which topics are mentioned in the text
        text_lower = text.lower()
        mentioned_topics = []
        
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                mentioned_topics.append(topic)
        
        return mentioned_topics


class CompletenessReport(BaseModel):
    """Report on the completeness of a response."""
    is_complete: bool = Field(..., description="Whether the response is complete")
    addressed_questions: List[Question] = Field(default_factory=list, description="Questions that were addressed")
    unanswered_questions: List[Question] = Field(default_factory=list, description="Questions that were not addressed")
    partially_addressed_questions: List[Question] = Field(default_factory=list, description="Questions that were partially addressed")
    missing_topics: List[str] = Field(default_factory=list, description="Topics that were not addressed")
    missing_entities: List[str] = Field(default_factory=list, description="Entities that were not addressed")
    completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="Score indicating how complete the response is")
    
    @property
    def total_questions(self) -> int:
        """Get the total number of questions."""
        return len(self.addressed_questions) + len(self.unanswered_questions) + len(self.partially_addressed_questions)
    
    @property
    def addressed_ratio(self) -> float:
        """Get the ratio of addressed questions."""
        if self.total_questions == 0:
            return 1.0
        return len(self.addressed_questions) / self.total_questions
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the completeness report."""
        return {
            "is_complete": self.is_complete,
            "total_questions": self.total_questions,
            "addressed_questions": len(self.addressed_questions),
            "unanswered_questions": len(self.unanswered_questions),
            "partially_addressed_questions": len(self.partially_addressed_questions),
            "completeness_score": self.completeness_score,
            "missing_topics": self.missing_topics,
            "missing_entities": self.missing_entities
        }


class CompletenessChecker(BaseModel):
    """
    System that verifies all parts of a multi-part question are addressed in the response.
    """
    min_entity_coverage: float = Field(0.5, ge=0.0, le=1.0, description="Minimum ratio of entities that must be covered")
    min_topic_coverage: float = Field(0.7, ge=0.0, le=1.0, description="Minimum ratio of topics that must be covered")
    
    def check_completeness(self, query: UserQuery, response: str) -> CompletenessReport:
        """
        Check if the response addresses all parts of the query.
        
        Args:
            query: The user query to check
            response: The agent response to validate
            
        Returns:
            CompletenessReport with the results of the check
        """
        # Normalize response
        normalized_response = response.lower()
        
        # Initialize report
        report = CompletenessReport(
            is_complete=True,
            addressed_questions=[],
            unanswered_questions=[],
            partially_addressed_questions=[],
            missing_topics=[],
            missing_entities=[],
            completeness_score=0.0
        )
        
        # If no questions, the response is complete by default
        if not query.questions:
            report.completeness_score = 1.0
            return report
        
        # Check each question
        all_entities = set()
        all_topics = set()
        
        for question in query.questions:
            # Add entities and topics to the overall sets
            all_entities.update(question.key_entities)
            all_topics.update(question.key_topics)
            
            # Check if the question is addressed
            is_addressed, coverage_score = self._is_question_addressed(question, normalized_response)
            
            if is_addressed:
                question.is_addressed = True
                report.addressed_questions.append(question)
            elif coverage_score > 0:
                question.is_addressed = False
                report.partially_addressed_questions.append(question)
            else:
                question.is_addressed = False
                report.unanswered_questions.append(question)
            
            # Check sub-questions if this is a multi-part question
            if question.is_multi_part:
                for sub_question in question.sub_questions:
                    sub_is_addressed, sub_coverage_score = self._is_question_addressed(sub_question, normalized_response)
                    
                    if sub_is_addressed:
                        sub_question.is_addressed = True
                    else:
                        sub_question.is_addressed = False
                        # If any sub-question is not addressed, the main question is at most partially addressed
                        if question in report.addressed_questions:
                            report.addressed_questions.remove(question)
                            report.partially_addressed_questions.append(question)
        
        # Check overall entity and topic coverage
        covered_entities = self._get_covered_entities(all_entities, normalized_response)
        covered_topics = self._get_covered_topics(all_topics, normalized_response)
        
        entity_coverage = len(covered_entities) / len(all_entities) if all_entities else 1.0
        topic_coverage = len(covered_topics) / len(all_topics) if all_topics else 1.0
        
        # Calculate missing entities and topics
        report.missing_entities = list(all_entities - covered_entities)
        report.missing_topics = list(all_topics - covered_topics)
        
        # Calculate completeness score
        question_score = report.addressed_ratio
        coverage_score = (entity_coverage + topic_coverage) / 2
        report.completeness_score = (question_score + coverage_score) / 2
        
        # Determine if the response is complete
        report.is_complete = (
            len(report.unanswered_questions) == 0 and
            entity_coverage >= self.min_entity_coverage and
            topic_coverage >= self.min_topic_coverage
        )
        
        return report
    
    def _is_question_addressed(self, question: Question, normalized_response: str) -> Tuple[bool, float]:
        """
        Check if a question is addressed in the response.
        
        Args:
            question: The question to check
            normalized_response: The normalized response text
            
        Returns:
            Tuple of (is_addressed, coverage_score)
        """
        # Check entity coverage
        entity_coverage = self._calculate_entity_coverage(question.key_entities, normalized_response)
        
        # Check topic coverage
        topic_coverage = self._calculate_topic_coverage(question.key_topics, normalized_response)
        
        # Check for question type specific patterns
        type_coverage = self._check_question_type_coverage(question.question_type, normalized_response)
        
        # Calculate overall coverage score
        coverage_score = (entity_coverage + topic_coverage + type_coverage) / 3
        
        # Question is addressed if coverage score is high enough
        is_addressed = coverage_score >= 0.7
        
        return is_addressed, coverage_score
    
    def _calculate_entity_coverage(self, entities: List[str], normalized_response: str) -> float:
        """Calculate what portion of entities are covered in the response."""
        if not entities:
            return 1.0
        
        covered_entities = sum(1 for entity in entities if entity.lower() in normalized_response)
        return covered_entities / len(entities)
    
    def _calculate_topic_coverage(self, topics: List[str], normalized_response: str) -> float:
        """Calculate what portion of topics are covered in the response."""
        if not topics:
            return 1.0
        
        # Define topic keywords
        topic_keywords = {
            "weather": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", "storm", "humidity", "wind"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel", "booking", "destination", "tourism", "visit"],
            "food": ["food", "restaurant", "recipe", "cooking", "meal", "dinner", "lunch", "breakfast", "ingredient"],
            "technology": ["technology", "computer", "software", "hardware", "app", "device", "digital", "internet", "online"],
            "health": ["health", "medical", "doctor", "symptom", "disease", "treatment", "medicine", "wellness", "fitness"],
            "finance": ["finance", "money", "bank", "investment", "stock", "budget", "saving", "expense", "financial"],
            "education": ["education", "school", "university", "college", "course", "study", "learning", "student", "teacher"],
            "entertainment": ["entertainment", "movie", "music", "game", "show", "concert", "theater", "play", "performance"]
        }
        
        covered_topics = 0
        for topic in topics:
            if topic in topic_keywords:
                # Check if any of the topic keywords are in the response
                if any(keyword in normalized_response for keyword in topic_keywords[topic]):
                    covered_topics += 1
            else:
                # If the topic itself is in the response, count it as covered
                if topic.lower() in normalized_response:
                    covered_topics += 1
        
        return covered_topics / len(topics)
    
    def _check_question_type_coverage(self, question_type: QuestionType, normalized_response: str) -> float:
        """Check if the response addresses the specific type of question."""
        # Define patterns for different question types
        type_patterns = {
            QuestionType.FACTUAL: [r'\b(is|are|was|were|has|have|had|can|could|will|would)\b'],
            QuestionType.TEMPORAL: [r'\b(time|date|when|year|month|day|hour|minute|second|period|duration)\b'],
            QuestionType.PROCEDURAL: [r'\b(first|then|next|finally|step|process|procedure|how to|method|way)\b'],
            QuestionType.CAUSAL: [r'\b(because|since|as|due to|result of|cause|reason|why|therefore|thus)\b'],
            QuestionType.COMPARATIVE: [r'\b(more|less|better|worse|higher|lower|than|compare|comparison|difference|similar|different)\b'],
            QuestionType.HYPOTHETICAL: [r'\b(if|would|could|might|may|possible|potentially|scenario|situation|case)\b'],
            QuestionType.PREFERENCE: [r'\b(prefer|recommend|suggest|advise|best|ideal|optimal|favorite|choice|option)\b'],
            QuestionType.CLARIFICATION: [r'\b(mean|meaning|definition|defined as|refers to|explanation|clarification|specifically)\b'],
            QuestionType.CONFIRMATION: [r'\b(yes|no|correct|incorrect|right|wrong|true|false|confirm|deny)\b'],
            QuestionType.QUANTITATIVE: [r'\b(number|amount|quantity|many|much|few|several|count|total|sum)\b']
        }
        
        # Check if any of the patterns for this question type are in the response
        patterns = type_patterns.get(question_type, [])
        for pattern in patterns:
            if re.search(pattern, normalized_response):
                return 1.0
        
        return 0.0
    
    def _get_covered_entities(self, entities: Set[str], normalized_response: str) -> Set[str]:
        """Get the set of entities that are covered in the response."""
        return {entity for entity in entities if entity.lower() in normalized_response}
    
    def _get_covered_topics(self, topics: Set[str], normalized_response: str) -> Set[str]:
        """Get the set of topics that are covered in the response."""
        # Define topic keywords
        topic_keywords = {
            "weather": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy", "storm", "humidity", "wind"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel", "booking", "destination", "tourism", "visit"],
            "food": ["food", "restaurant", "recipe", "cooking", "meal", "dinner", "lunch", "breakfast", "ingredient"],
            "technology": ["technology", "computer", "software", "hardware", "app", "device", "digital", "internet", "online"],
            "health": ["health", "medical", "doctor", "symptom", "disease", "treatment", "medicine", "wellness", "fitness"],
            "finance": ["finance", "money", "bank", "investment", "stock", "budget", "saving", "expense", "financial"],
            "education": ["education", "school", "university", "college", "course", "study", "learning", "student", "teacher"],
            "entertainment": ["entertainment", "movie", "music", "game", "show", "concert", "theater", "play", "performance"]
        }
        
        covered = set()
        for topic in topics:
            if topic in topic_keywords:
                # Check if any of the topic keywords are in the response
                if any(keyword in normalized_response for keyword in topic_keywords[topic]):
                    covered.add(topic)
            else:
                # If the topic itself is in the response, count it as covered
                if topic.lower() in normalized_response:
                    covered.add(topic)
        
        return covered


# Example usage
if __name__ == "__main__":
    # Example 1: Simple question
    query1 = UserQuery(text="What's the weather like in New York?")
    response1 = "The weather in New York is currently sunny with a temperature of 75°F."
    
    checker = CompletenessChecker()
    report1 = checker.check_completeness(query1, response1)
    
    print("Example 1: Simple question")
    print(f"Query: '{query1.text}'")
    print(f"Response: '{response1}'")
    print(f"Is complete: {report1.is_complete}")
    print(f"Completeness score: {report1.completeness_score:.2f}")
    print(f"Questions extracted: {len(query1.questions)}")
    for i, question in enumerate(query1.questions):
        print(f"  Question {i+1}: '{question.text}'")
        print(f"    Type: {question.question_type}")
        print(f"    Key entities: {question.key_entities}")
        print(f"    Key topics: {question.key_topics}")
        print(f"    Is addressed: {question.is_addressed}")
    print()
    
    # Example 2: Multi-part question
    query2 = UserQuery(text="What's the weather like in New York? And what's the best time to visit?")
    response2 = "The weather in New York is currently sunny with a temperature of 75°F. The best time to visit New York is during spring (April to June) or fall (September to November) when the weather is mild."
    
    report2 = checker.check_completeness(query2, response2)
    
    print("Example 2: Multi-part question")
    print(f"Query: '{query2.text}'")
    print(f"Response: '{response2}'")
    print(f"Is complete: {report2.is_complete}")
    print(f"Completeness score: {report2.completeness_score:.2f}")
    print(f"Questions extracted: {len(query2.questions)}")
    for i, question in enumerate(query2.questions):
        print(f"  Question {i+1}: '{question.text}'")
        print(f"    Type: {question.question_type}")
        print(f"    Key entities: {question.key_entities}")
        print(f"    Key topics: {question.key_topics}")
        print(f"    Is addressed: {question.is_addressed}")
        if question.is_multi_part:
            print(f"    Sub-questions: {len(question.sub_questions)}")
            for j, sub_question in enumerate(question.sub_questions):
                print(f"      Sub-question {j+1}: '{sub_question.text}'")
                print(f"        Type: {sub_question.question_type}")
                print(f"        Is addressed: {sub_question.is_addressed}")
    print()
    
    # Example 3: Incomplete response
    query3 = UserQuery(text="What's the weather like in New York? And what's the best time to visit?")
    response3 = "The weather in New York is currently sunny with a temperature of 75°F."
    
    report3 = checker.check_completeness(query3, response3)
    
    print("Example 3: Incomplete response")
    print(f"Query: '{query3.text}'")
    print(f"Response: '{response3}'")
    print(f"Is complete: {report3.is_complete}")
    print(f"Completeness score: {report3.completeness_score:.2f}")
    print(f"Addressed questions: {len(report3.addressed_questions)}")
    print(f"Partially addressed questions: {len(report3.partially_addressed_questions)}")
    print(f"Unanswered questions: {len(report3.unanswered_questions)}")
    if report3.unanswered_questions:
        print("Unanswered questions:")
        for question in report3.unanswered_questions:
            print(f"  - '{question.text}'")
    print(f"Missing topics: {report3.missing_topics}")
    print(f"Missing entities: {report3.missing_entities}")
    print()
    
    # Example 4: Complex multi-part query
    query4 = UserQuery(text="Can you tell me about the weather in New York, the best restaurants to visit, and how to get around the city? Also, what are some free activities for tourists?")
    response4 = "New York currently has sunny weather with temperatures around 75°F. For restaurants, I recommend trying the famous pizza at Joe's Pizza or fine dining at Per Se. To get around, the subway is the most efficient option, with a 7-day unlimited MetroCard costing $33. As for free activities, you can visit Central Park, walk the High Line, or check out the Staten Island Ferry for great views of the Statue of Liberty."
    
    report4 = checker.check_completeness(query4, response4)
    
    print("Example 4: Complex multi-part query")
    print(f"Query: '{query4.text}'")
    print(f"Response: '{response4}'")
    print(f"Is complete: {report4.is_complete}")
    print(f"Completeness score: {report4.completeness_score:.2f}")
    print(f"Questions extracted: {len(query4.questions)}")
    for i, question in enumerate(query4.questions):
        print(f"  Question {i+1}: '{question.text}'")
        print(f"    Type: {question.question_type}")
        print(f"    Key entities: {question.key_entities}")
        print(f"    Key topics: {question.key_topics}")
        print(f"    Is addressed: {question.is_addressed}")
    print(f"Report summary: {report4.get_summary()}")

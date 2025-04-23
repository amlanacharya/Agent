"""
Lesson 3 Exercise Solutions
-------------------------
This module contains solutions for the exercises in Lesson 3: Structured Output Parsing.
"""

import json
import re
from typing import List, Optional, Dict, Any, Union, Callable
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# Exercise 1: Create a Pydantic model for a job application form with fields for
# personal information, education, work experience, and skills.

class Education(BaseModel):
    """Educational background information."""
    institution: str = Field(..., description="Name of the educational institution")
    degree: str = Field(..., description="Degree or certification obtained")
    field_of_study: str = Field(..., description="Field or major of study")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format (or None if ongoing)")
    gpa: Optional[float] = Field(None, description="Grade Point Average (if applicable)")
    
    @field_validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Validate that dates are in YYYY-MM-DD format."""
        if v is None:
            return v
        
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        
        # Additional validation for valid date
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date")
        
        return v
    
    @field_validator('end_date')
    def validate_end_date_after_start_date(cls, v, info):
        """Validate that end_date is after start_date if provided."""
        if v is None:
            return v
        
        if 'start_date' in info.data:
            start_date = datetime.strptime(info.data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(v, '%Y-%m-%d')
            
            if end_date < start_date:
                raise ValueError("End date must be after start date")
        
        return v


class WorkExperience(BaseModel):
    """Work experience information."""
    company: str = Field(..., description="Name of the company or organization")
    position: str = Field(..., description="Job title or position")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format (or None if current)")
    is_current: bool = Field(False, description="Whether this is the current position")
    responsibilities: List[str] = Field(..., description="List of job responsibilities")
    achievements: Optional[List[str]] = Field(None, description="List of achievements or accomplishments")
    
    @field_validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Validate that dates are in YYYY-MM-DD format."""
        if v is None:
            return v
        
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError("Date must be in YYYY-MM-DD format")
        
        # Additional validation for valid date
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Invalid date")
        
        return v
    
    @field_validator('end_date')
    def validate_end_date_after_start_date(cls, v, info):
        """Validate that end_date is after start_date if provided."""
        if v is None:
            return v
        
        if 'start_date' in info.data:
            start_date = datetime.strptime(info.data['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(v, '%Y-%m-%d')
            
            if end_date < start_date:
                raise ValueError("End date must be after start date")
        
        return v
    
    @field_validator('end_date')
    def validate_end_date_with_is_current(cls, v, info):
        """Validate that end_date is None if is_current is True."""
        if 'is_current' in info.data and info.data['is_current'] and v is not None:
            raise ValueError("End date must be None for current positions")
        
        return v


class Skill(BaseModel):
    """Skill information."""
    name: str = Field(..., description="Name of the skill")
    level: str = Field(..., description="Proficiency level (e.g., Beginner, Intermediate, Advanced, Expert)")
    years_experience: Optional[int] = Field(None, description="Years of experience with this skill")
    
    @field_validator('level')
    def validate_level(cls, v):
        """Validate that level is one of the allowed values."""
        allowed_levels = ["beginner", "intermediate", "advanced", "expert"]
        if v.lower() not in allowed_levels:
            raise ValueError(f"Level must be one of: {', '.join(allowed_levels)}")
        
        return v.lower()


class JobApplication(BaseModel):
    """Complete job application form."""
    # Personal Information
    full_name: str = Field(..., description="Applicant's full name")
    email: str = Field(..., description="Applicant's email address")
    phone: str = Field(..., description="Applicant's phone number")
    address: Optional[str] = Field(None, description="Applicant's physical address")
    linkedin_url: Optional[str] = Field(None, description="Applicant's LinkedIn profile URL")
    github_url: Optional[str] = Field(None, description="Applicant's GitHub profile URL")
    portfolio_url: Optional[str] = Field(None, description="Applicant's portfolio website URL")
    
    # Education and Experience
    education: List[Education] = Field(..., description="List of educational backgrounds")
    work_experience: List[WorkExperience] = Field(..., description="List of work experiences")
    skills: List[Skill] = Field(..., description="List of skills")
    
    # Additional Information
    summary: str = Field(..., description="Brief professional summary or objective")
    cover_letter: Optional[str] = Field(None, description="Cover letter content")
    references: Optional[List[Dict[str, str]]] = Field(None, description="List of professional references")
    availability: Optional[str] = Field(None, description="Availability to start work")
    
    @field_validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator('phone')
    def validate_phone(cls, v):
        """Validate phone number format."""
        if not re.match(r"^\+?[\d\s\-\(\)]+$", v):
            raise ValueError("Invalid phone number format")
        return v
    
    @field_validator('linkedin_url', 'github_url', 'portfolio_url')
    def validate_url(cls, v):
        """Validate URL format if provided."""
        if v is None:
            return v
        
        if not re.match(r"^https?://", v):
            raise ValueError("URL must start with http:// or https://")
        
        return v


# Exercise 2: Implement a PydanticOutputParser for your job application model
# and test it with sample text.

class OutputParser:
    """Base class for output parsers."""
    
    def parse(self, text: str) -> Any:
        """
        Parse text into structured data.
        
        Args:
            text: Text to parse
            
        Returns:
            Parsed data
            
        Raises:
            ValueError: If text cannot be parsed
        """
        raise NotImplementedError("Subclasses must implement parse method")
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for the LLM on how to format its output.
        
        Returns:
            Formatting instructions as a string
        """
        raise NotImplementedError("Subclasses must implement get_format_instructions method")


class JobApplicationParser(OutputParser):
    """Parser for job application data."""
    
    def __init__(self):
        """Initialize the parser."""
        self.model_class = JobApplication
    
    def parse(self, text: str) -> JobApplication:
        """
        Parse text into a JobApplication model.
        
        Args:
            text: Text to parse
            
        Returns:
            JobApplication instance
            
        Raises:
            ValueError: If text cannot be parsed into a JobApplication
        """
        try:
            # Try to find JSON in the output
            json_data = self._extract_json(text)
            
            # Validate with Pydantic model
            return self.model_class(**json_data)
        except Exception as e:
            raise ValueError(f"Could not parse text as JobApplication: {str(e)}")
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON as a dictionary
            
        Raises:
            ValueError: If JSON cannot be extracted
        """
        # Try to find JSON in code blocks
        code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except:
                pass
        
        # Try to find JSON-like structure
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
        except:
            pass
        
        # Try to fix common JSON errors
        try:
            # Replace single quotes with double quotes
            fixed_json = text.replace("'", '"')
            start_idx = fixed_json.find('{')
            end_idx = fixed_json.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = fixed_json[start_idx:end_idx+1]
                return json.loads(json_str)
        except:
            pass
        
        # If all parsing attempts fail
        raise ValueError("Could not extract JSON from text")
    
    def get_format_instructions(self) -> str:
        """
        Get instructions for the LLM on how to format its output.
        
        Returns:
            Formatting instructions as a string
        """
        schema = self.model_class.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        return f"""
        You must format your output as a JSON instance that conforms to the JSON schema below.

        {schema_str}
        
        The JSON output should contain only the required fields and should be valid JSON.
        Make sure to format dates as YYYY-MM-DD strings.
        """


# Exercise 3: Add custom validators to ensure that dates are in the correct format
# and email addresses are valid.
# (Already implemented in the models above)


# Exercise 4: Implement a retry mechanism that provides more specific instructions
# when parsing fails.

def parse_with_retry(
    llm_call: Callable[[str], str],
    parser: JobApplicationParser,
    text: str,
    max_retries: int = 3
) -> JobApplication:
    """
    Try to parse job application data, retrying with more specific instructions if it fails.
    
    Args:
        llm_call: Function that calls the LLM with a prompt
        parser: Parser to use for parsing the output
        text: Text to process
        max_retries: Maximum number of retry attempts
        
    Returns:
        JobApplication instance
        
    Raises:
        ValueError: If parsing fails after all retries
    """
    prompt_template = """
    Extract job application information from the text below:
    
    {text}
    
    {format_instructions}
    
    {retry_instructions}
    """
    
    retry_instructions = ""
    
    for attempt in range(max_retries):
        # Format the prompt
        prompt = prompt_template.format(
            text=text,
            format_instructions=parser.get_format_instructions(),
            retry_instructions=retry_instructions
        )
        
        # Call the LLM
        output = llm_call(prompt)
        
        try:
            return parser.parse(output)
        except Exception as e:
            if attempt < max_retries - 1:
                # Add more specific instructions for the next attempt
                retry_instructions = f"""
                The previous response could not be parsed correctly. Error: {e}
                
                Please make sure your response strictly follows the format instructions.
                Double-check that:
                1. All required fields are included
                2. The JSON format is valid
                3. Dates are in YYYY-MM-DD format
                4. Email addresses are valid
                5. Phone numbers are valid
                6. URLs start with http:// or https://
                7. Skill levels are one of: beginner, intermediate, advanced, expert
                """
            else:
                raise ValueError(f"Failed to parse job application after {max_retries} attempts: {e}")


# Exercise 5: Create a two-stage parsing approach for complex job applications
# with nested information.

def two_stage_parsing(
    llm_call: Callable[[str], str],
    text: str
) -> JobApplication:
    """
    Use a two-stage approach to parse complex job application data.
    
    Args:
        llm_call: Function that calls the LLM with a prompt
        text: Text to process
        
    Returns:
        JobApplication instance
        
    Raises:
        ValueError: If parsing fails
    """
    # Stage 1: Extract basic information
    basic_prompt = """
    Extract the following basic information from the job application:
    
    1. Personal Information (full_name, email, phone)
    2. Summary or objective statement
    3. Number of education entries
    4. Number of work experience entries
    5. List of skills
    
    Text: {text}
    
    Provide the information in JSON format.
    """
    
    initial_output = llm_call(basic_prompt.format(text=text))
    
    try:
        basic_info = json.loads(initial_output)
    except:
        # Try to extract JSON from the output
        try:
            start_idx = initial_output.find('{')
            end_idx = initial_output.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = initial_output[start_idx:end_idx+1]
                basic_info = json.loads(json_str)
            else:
                raise ValueError("Could not extract JSON from initial output")
        except:
            raise ValueError("Could not parse initial output")
    
    # Stage 2: Extract detailed information
    detailed_prompt = """
    I've extracted the following basic information from the job application:
    
    {basic_info}
    
    Now, please extract the complete job application information, including:
    
    1. All personal information (full_name, email, phone, address, linkedin_url, github_url, portfolio_url)
    2. Education details for each education entry ({education_count} entries)
    3. Work experience details for each work experience entry ({work_experience_count} entries)
    4. Detailed skill information for each skill
    5. Summary, cover letter, references, and availability
    
    Text: {text}
    
    Format the output as a valid JSON object that conforms to the following schema:
    
    {schema}
    
    Ensure all required fields are present and correctly formatted.
    """
    
    # Get education and work experience counts
    education_count = basic_info.get("education_count", 0)
    work_experience_count = basic_info.get("work_experience_count", 0)
    
    # Get schema
    schema = JobApplication.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    
    # Format the prompt
    detailed_prompt = detailed_prompt.format(
        basic_info=json.dumps(basic_info, indent=2),
        education_count=education_count,
        work_experience_count=work_experience_count,
        text=text,
        schema=schema_str
    )
    
    # Call the LLM
    detailed_output = llm_call(detailed_prompt)
    
    # Parse the output
    parser = JobApplicationParser()
    return parser.parse(detailed_output)


# Simulated LLM for testing
# -----------------------

def simulate_llm_call(prompt: str) -> str:
    """
    Simulate an LLM call for testing purposes.
    
    Args:
        prompt: Prompt to send to the LLM
        
    Returns:
        Simulated LLM response
    """
    # This is a very simple simulation that just returns predefined responses
    # In a real application, this would call an actual LLM API
    
    if "job application" in prompt.lower() and "john smith" in prompt.lower():
        return """
        ```json
        {
          "full_name": "John Smith",
          "email": "john.smith@example.com",
          "phone": "+1 (555) 123-4567",
          "address": "123 Main St, Anytown, CA 12345",
          "linkedin_url": "https://linkedin.com/in/johnsmith",
          "github_url": "https://github.com/johnsmith",
          "portfolio_url": "https://johnsmith.dev",
          "education": [
            {
              "institution": "University of Example",
              "degree": "Bachelor of Science",
              "field_of_study": "Computer Science",
              "start_date": "2015-09-01",
              "end_date": "2019-05-15",
              "gpa": 3.8
            },
            {
              "institution": "Example Technical Institute",
              "degree": "Master of Science",
              "field_of_study": "Software Engineering",
              "start_date": "2019-09-01",
              "end_date": "2021-05-15",
              "gpa": 3.9
            }
          ],
          "work_experience": [
            {
              "company": "Tech Innovations Inc.",
              "position": "Software Engineer",
              "start_date": "2021-06-01",
              "end_date": null,
              "is_current": true,
              "responsibilities": [
                "Develop and maintain web applications",
                "Collaborate with cross-functional teams",
                "Implement new features and fix bugs"
              ],
              "achievements": [
                "Reduced application load time by 40%",
                "Implemented CI/CD pipeline"
              ]
            },
            {
              "company": "Code Solutions LLC",
              "position": "Junior Developer",
              "start_date": "2019-05-20",
              "end_date": "2021-05-30",
              "is_current": false,
              "responsibilities": [
                "Assisted in web application development",
                "Fixed bugs and implemented minor features",
                "Participated in code reviews"
              ],
              "achievements": [
                "Developed a utility library used across projects",
                "Received 'Rookie of the Year' award"
              ]
            }
          ],
          "skills": [
            {
              "name": "Python",
              "level": "expert",
              "years_experience": 5
            },
            {
              "name": "JavaScript",
              "level": "advanced",
              "years_experience": 4
            },
            {
              "name": "React",
              "level": "intermediate",
              "years_experience": 2
            }
          ],
          "summary": "Experienced software engineer with a strong background in web development and a passion for creating efficient, scalable applications.",
          "cover_letter": "Dear Hiring Manager,\\n\\nI am writing to express my interest in the Software Engineer position at your company...",
          "references": [
            {
              "name": "Jane Doe",
              "position": "Senior Developer",
              "company": "Tech Innovations Inc.",
              "email": "jane.doe@example.com",
              "phone": "+1 (555) 987-6543"
            }
          ],
          "availability": "Available to start immediately"
        }
        ```
        """
    
    if "basic information" in prompt.lower():
        return """
        {
          "full_name": "John Smith",
          "email": "john.smith@example.com",
          "phone": "+1 (555) 123-4567",
          "summary": "Experienced software engineer with a strong background in web development",
          "education_count": 2,
          "work_experience_count": 2,
          "skills": ["Python", "JavaScript", "React"]
        }
        """
    
    # Default response for unknown prompts
    return """
    {
      "error": "I don't have enough information to provide a structured response."
    }
    """


# Demonstration functions
# ---------------------

def demonstrate_job_application_parser():
    """Demonstrate JobApplicationParser."""
    # Create parser
    parser = JobApplicationParser()
    
    # Get format instructions
    instructions = parser.get_format_instructions()
    print(f"Format instructions (excerpt):\n{instructions[:500]}...\n")
    
    # Simulate LLM call with instructions
    prompt = f"""
    Extract job application information from this text:
    
    John Smith is applying for a Software Engineer position. He has a Bachelor's degree in Computer Science
    from the University of Example (2015-2019) and a Master's in Software Engineering from Example Technical
    Institute (2019-2021). He currently works at Tech Innovations Inc. as a Software Engineer since June 2021,
    and previously worked at Code Solutions LLC as a Junior Developer from May 2019 to May 2021.
    
    His email is john.smith@example.com and phone number is +1 (555) 123-4567.
    
    {instructions}
    """
    
    llm_output = simulate_llm_call(prompt)
    print(f"LLM output (excerpt):\n{llm_output[:500]}...\n")
    
    # Parse the output
    try:
        job_application = parser.parse(llm_output)
        print(f"Parsed Job Application:")
        print(f"  Name: {job_application.full_name}")
        print(f"  Email: {job_application.email}")
        print(f"  Education: {len(job_application.education)} entries")
        print(f"  Work Experience: {len(job_application.work_experience)} entries")
        print(f"  Skills: {len(job_application.skills)} entries")
    except Exception as e:
        print(f"Parsing error: {e}")


def demonstrate_retry_mechanism():
    """Demonstrate retry mechanism."""
    # Create parser
    parser = JobApplicationParser()
    
    # Mock LLM call function that succeeds on the second attempt
    attempt_count = [0]
    
    def mock_llm_call(prompt: str) -> str:
        attempt_count[0] += 1
        
        if attempt_count[0] == 1:
            # First attempt returns invalid JSON
            return "This is not valid JSON"
        else:
            # Second attempt returns valid JSON
            return simulate_llm_call("job application john smith")
    
    # Test retry logic
    try:
        job_application = parse_with_retry(
            llm_call=mock_llm_call,
            parser=parser,
            text="John Smith is applying for a Software Engineer position...",
            max_retries=3
        )
        
        print(f"Parsed Job Application after {attempt_count[0]} attempts:")
        print(f"  Name: {job_application.full_name}")
        print(f"  Email: {job_application.email}")
        print(f"  Education: {len(job_application.education)} entries")
        print(f"  Work Experience: {len(job_application.work_experience)} entries")
        print(f"  Skills: {len(job_application.skills)} entries")
    except Exception as e:
        print(f"Parsing error: {e}")


def demonstrate_two_stage_parsing():
    """Demonstrate two-stage parsing."""
    try:
        job_application = two_stage_parsing(
            llm_call=simulate_llm_call,
            text="John Smith is applying for a Software Engineer position..."
        )
        
        print(f"Parsed Job Application using two-stage parsing:")
        print(f"  Name: {job_application.full_name}")
        print(f"  Email: {job_application.email}")
        print(f"  Education: {len(job_application.education)} entries")
        print(f"  Work Experience: {len(job_application.work_experience)} entries")
        print(f"  Skills: {len(job_application.skills)} entries")
    except Exception as e:
        print(f"Parsing error: {e}")


if __name__ == "__main__":
    print("=== Job Application Parser ===")
    demonstrate_job_application_parser()
    
    print("\n=== Retry Mechanism ===")
    demonstrate_retry_mechanism()
    
    print("\n=== Two-Stage Parsing ===")
    demonstrate_two_stage_parsing()

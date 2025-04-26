"""
Tests for Lesson 3 Exercise Solutions
----------------------------------
This module contains tests for the lesson3_exercises module.
"""

import unittest
from datetime import datetime
from pydantic import ValidationError

from module3.exercises.lesson3_exercises import (
    Education,
    WorkExperience,
    Skill,
    JobApplication,
    JobApplicationParser,
    parse_with_retry,
    two_stage_parsing,
    simulate_llm_call
)


class TestLesson3Exercises(unittest.TestCase):
    """Test cases for lesson3_exercises module."""

    def test_education_model(self):
        """Test the Education model."""
        # Valid education
        education = Education(
            institution="University of Example",
            degree="Bachelor of Science",
            field_of_study="Computer Science",
            start_date="2015-09-01",
            end_date="2019-05-15",
            gpa=3.8
        )

        self.assertEqual(education.institution, "University of Example")
        self.assertEqual(education.degree, "Bachelor of Science")
        self.assertEqual(education.field_of_study, "Computer Science")
        self.assertEqual(education.start_date, "2015-09-01")
        self.assertEqual(education.end_date, "2019-05-15")
        self.assertEqual(education.gpa, 3.8)

        # Test with ongoing education (no end date)
        ongoing_education = Education(
            institution="University of Example",
            degree="Master of Science",
            field_of_study="Computer Science",
            start_date="2019-09-01"
        )

        self.assertEqual(ongoing_education.institution, "University of Example")
        self.assertIsNone(ongoing_education.end_date)

        # Test invalid date format
        with self.assertRaises(ValidationError):
            Education(
                institution="University of Example",
                degree="Bachelor of Science",
                field_of_study="Computer Science",
                start_date="09/01/2015",  # Invalid format
                end_date="2019-05-15"
            )

        # Test end date before start date
        with self.assertRaises(ValidationError):
            Education(
                institution="University of Example",
                degree="Bachelor of Science",
                field_of_study="Computer Science",
                start_date="2019-09-01",
                end_date="2015-05-15"  # Before start date
            )

    def test_work_experience_model(self):
        """Test the WorkExperience model."""
        # Valid work experience
        experience = WorkExperience(
            company="Tech Innovations Inc.",
            position="Software Engineer",
            start_date="2021-06-01",
            end_date="2023-05-30",
            is_current=False,
            responsibilities=[
                "Develop and maintain web applications",
                "Collaborate with cross-functional teams"
            ],
            achievements=[
                "Reduced application load time by 40%",
                "Implemented CI/CD pipeline"
            ]
        )

        self.assertEqual(experience.company, "Tech Innovations Inc.")
        self.assertEqual(experience.position, "Software Engineer")
        self.assertEqual(experience.start_date, "2021-06-01")
        self.assertEqual(experience.end_date, "2023-05-30")
        self.assertFalse(experience.is_current)
        self.assertEqual(len(experience.responsibilities), 2)
        self.assertEqual(len(experience.achievements), 2)

        # Test current position
        current_experience = WorkExperience(
            company="Tech Innovations Inc.",
            position="Software Engineer",
            start_date="2021-06-01",
            is_current=True,
            responsibilities=[
                "Develop and maintain web applications"
            ]
        )

        self.assertEqual(current_experience.company, "Tech Innovations Inc.")
        self.assertIsNone(current_experience.end_date)
        self.assertTrue(current_experience.is_current)

        # Test invalid date format
        with self.assertRaises(ValidationError):
            WorkExperience(
                company="Tech Innovations Inc.",
                position="Software Engineer",
                start_date="06/01/2021",  # Invalid format
                responsibilities=[
                    "Develop and maintain web applications"
                ]
            )

        # Test end date with current position
        # Create a work experience with is_current=True but with an end_date
        work_exp = WorkExperience(
            company="Tech Innovations Inc.",
            position="Software Engineer",
            start_date="2021-06-01",
            end_date="2023-05-30",  # Should be None for current position
            is_current=True,
            responsibilities=[
                "Develop and maintain web applications"
            ]
        )

        # Verify that the model_post_init method corrected the inconsistency
        self.assertIsNone(work_exp.end_date)

    def test_skill_model(self):
        """Test the Skill model."""
        # Valid skill
        skill = Skill(
            name="Python",
            level="expert",
            years_experience=5
        )

        self.assertEqual(skill.name, "Python")
        self.assertEqual(skill.level, "expert")
        self.assertEqual(skill.years_experience, 5)

        # Test with capitalized level
        skill = Skill(
            name="JavaScript",
            level="Advanced",  # Should be normalized to lowercase
            years_experience=3
        )

        self.assertEqual(skill.name, "JavaScript")
        self.assertEqual(skill.level, "advanced")

        # Test without years_experience
        skill = Skill(
            name="React",
            level="intermediate"
        )

        self.assertEqual(skill.name, "React")
        self.assertEqual(skill.level, "intermediate")
        self.assertIsNone(skill.years_experience)

        # Test invalid level
        with self.assertRaises(ValidationError):
            Skill(
                name="Python",
                level="master"  # Invalid level
            )

    def test_job_application_model(self):
        """Test the JobApplication model."""
        # Valid job application
        job_application = JobApplication(
            full_name="John Smith",
            email="john.smith@example.com",
            phone="+1 (555) 123-4567",
            address="123 Main St, Anytown, CA 12345",
            linkedin_url="https://linkedin.com/in/johnsmith",
            github_url="https://github.com/johnsmith",
            portfolio_url="https://johnsmith.dev",
            education=[
                Education(
                    institution="University of Example",
                    degree="Bachelor of Science",
                    field_of_study="Computer Science",
                    start_date="2015-09-01",
                    end_date="2019-05-15",
                    gpa=3.8
                )
            ],
            work_experience=[
                WorkExperience(
                    company="Tech Innovations Inc.",
                    position="Software Engineer",
                    start_date="2021-06-01",
                    is_current=True,
                    responsibilities=[
                        "Develop and maintain web applications"
                    ]
                )
            ],
            skills=[
                Skill(
                    name="Python",
                    level="expert",
                    years_experience=5
                )
            ],
            summary="Experienced software engineer with a strong background in web development."
        )

        self.assertEqual(job_application.full_name, "John Smith")
        self.assertEqual(job_application.email, "john.smith@example.com")
        self.assertEqual(job_application.phone, "+1 (555) 123-4567")
        self.assertEqual(len(job_application.education), 1)
        self.assertEqual(len(job_application.work_experience), 1)
        self.assertEqual(len(job_application.skills), 1)

        # Test invalid email
        with self.assertRaises(ValidationError):
            JobApplication(
                full_name="John Smith",
                email="invalid-email",  # Invalid email
                phone="+1 (555) 123-4567",
                education=[
                    Education(
                        institution="University of Example",
                        degree="Bachelor of Science",
                        field_of_study="Computer Science",
                        start_date="2015-09-01",
                        end_date="2019-05-15"
                    )
                ],
                work_experience=[
                    WorkExperience(
                        company="Tech Innovations Inc.",
                        position="Software Engineer",
                        start_date="2021-06-01",
                        is_current=True,
                        responsibilities=[
                            "Develop and maintain web applications"
                        ]
                    )
                ],
                skills=[
                    Skill(
                        name="Python",
                        level="expert"
                    )
                ],
                summary="Experienced software engineer."
            )

        # Test invalid phone
        with self.assertRaises(ValidationError):
            JobApplication(
                full_name="John Smith",
                email="john.smith@example.com",
                phone="not-a-phone-number",  # Invalid phone
                education=[
                    Education(
                        institution="University of Example",
                        degree="Bachelor of Science",
                        field_of_study="Computer Science",
                        start_date="2015-09-01",
                        end_date="2019-05-15"
                    )
                ],
                work_experience=[
                    WorkExperience(
                        company="Tech Innovations Inc.",
                        position="Software Engineer",
                        start_date="2021-06-01",
                        is_current=True,
                        responsibilities=[
                            "Develop and maintain web applications"
                        ]
                    )
                ],
                skills=[
                    Skill(
                        name="Python",
                        level="expert"
                    )
                ],
                summary="Experienced software engineer."
            )

        # Test invalid URL
        with self.assertRaises(ValidationError):
            JobApplication(
                full_name="John Smith",
                email="john.smith@example.com",
                phone="+1 (555) 123-4567",
                linkedin_url="linkedin.com/in/johnsmith",  # Missing http:// or https://
                education=[
                    Education(
                        institution="University of Example",
                        degree="Bachelor of Science",
                        field_of_study="Computer Science",
                        start_date="2015-09-01",
                        end_date="2019-05-15"
                    )
                ],
                work_experience=[
                    WorkExperience(
                        company="Tech Innovations Inc.",
                        position="Software Engineer",
                        start_date="2021-06-01",
                        is_current=True,
                        responsibilities=[
                            "Develop and maintain web applications"
                        ]
                    )
                ],
                skills=[
                    Skill(
                        name="Python",
                        level="expert"
                    )
                ],
                summary="Experienced software engineer."
            )

    def test_job_application_parser(self):
        """Test the JobApplicationParser class."""
        # Create parser
        parser = JobApplicationParser()

        # Test format instructions
        instructions = parser.get_format_instructions()
        self.assertIn("JSON schema", instructions)
        self.assertIn("full_name", instructions)
        self.assertIn("email", instructions)
        self.assertIn("education", instructions)
        self.assertIn("work_experience", instructions)
        self.assertIn("skills", instructions)

        # Test parsing valid output
        valid_output = simulate_llm_call("job application john smith")

        job_application = parser.parse(valid_output)
        self.assertEqual(job_application.full_name, "John Smith")
        self.assertEqual(job_application.email, "john.smith@example.com")
        self.assertEqual(len(job_application.education), 2)
        self.assertEqual(len(job_application.work_experience), 2)
        self.assertEqual(len(job_application.skills), 3)

        # Test parsing invalid output
        with self.assertRaises(ValueError):
            parser.parse("This is not valid JSON")

    def test_simulate_llm_call(self):
        """Test the simulate_llm_call function."""
        # Test with job application prompt
        job_app_prompt = "Extract job application information for John Smith."
        job_app_output = simulate_llm_call(job_app_prompt)
        self.assertIn("John Smith", job_app_output)
        self.assertIn("john.smith@example.com", job_app_output)

        # Test with basic information prompt
        basic_prompt = "Extract basic information from the job application."
        basic_output = simulate_llm_call(basic_prompt)
        self.assertIn("John Smith", basic_output)
        self.assertIn("education_count", basic_output)

        # Test with unknown prompt
        unknown_prompt = "This is an unknown prompt."
        unknown_output = simulate_llm_call(unknown_prompt)
        self.assertIn("error", unknown_output)

    def test_parse_with_retry(self):
        """Test the parse_with_retry function."""
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
        job_application = parse_with_retry(
            llm_call=mock_llm_call,
            parser=parser,
            text="John Smith is applying for a Software Engineer position...",
            max_retries=3
        )

        self.assertEqual(job_application.full_name, "John Smith")
        self.assertEqual(job_application.email, "john.smith@example.com")
        self.assertEqual(len(job_application.education), 2)
        self.assertEqual(attempt_count[0], 2)  # Should succeed on second attempt

        # Test with always failing LLM call
        def failing_llm_call(prompt: str) -> str:
            return "This is not valid JSON"

        with self.assertRaises(ValueError):
            parse_with_retry(
                llm_call=failing_llm_call,
                parser=parser,
                text="John Smith is applying for a Software Engineer position...",
                max_retries=2
            )

    def test_two_stage_parsing(self):
        """Test the two_stage_parsing function."""
        # Test two-stage parsing
        job_application = two_stage_parsing(
            llm_call=simulate_llm_call,
            text="John Smith is applying for a Software Engineer position..."
        )

        self.assertEqual(job_application.full_name, "John Smith")
        self.assertEqual(job_application.email, "john.smith@example.com")
        self.assertEqual(len(job_application.education), 2)
        self.assertEqual(len(job_application.work_experience), 2)
        self.assertEqual(len(job_application.skills), 3)


if __name__ == "__main__":
    unittest.main()

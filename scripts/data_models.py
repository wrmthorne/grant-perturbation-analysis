from pydantic import BaseModel, Field
from typing import Literal


class Question(BaseModel):
    question_num: int
    question_title: str
    question_text: str
    response_text: str | None # None in the case of NA responses


class ApplicationOverview(BaseModel):
    title: str
    summary: str
    questions: list[Question]


class VisionAndApproach(BaseModel):
    title: str
    vision: str
    approach: str


class OpportunitySection(BaseModel):
    title: str
    content: str


class Opportunity(BaseModel):
    title: str
    metadata: dict
    sections: list[OpportunitySection]


class FitToOpportunity(BaseModel):
    title: str
    sections: list[Question]


class Review(BaseModel):
    score: int = Field(..., ge=1, le=6)
    explanation: str


class SectionComments(BaseModel):
    title: str
    comments: str


class SectionedReview(BaseModel):
    score: int = Field(..., ge=1, le=6)
    sections: list[SectionComments]


class PerturbationData(BaseModel):
    name: str
    description: str
    files: dict[str, str]
    diffs: dict[str, str]


class ProposalData(BaseModel):
    id: str
    original_files: dict[str, str]
    original_reviews: list[Review] | None
    perturbations: list[PerturbationData]


class PerturbationDetection(BaseModel):
    verdict: Literal["C", "P", "I"]
    explanation: str
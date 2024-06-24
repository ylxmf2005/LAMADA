import os
import regex as re
import json
import dspy
from dspy.predict import Retry
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
turbo = dspy.OpenAI(model='gpt-3.5-turbo-0613', max_tokens=500)
dspy.settings.configure(lm=turbo, trace=[], temperature=0.7)

dataset = HotPotQA(train_seed=1, train_size=300, eval_seed=2023, dev_size=300, test_size=0, keep_details=True)
trainset = [x.with_inputs('question', 'answer') for x in dataset.train]
devset = [x.with_inputs('question', 'answer') for x in dataset.dev]

class GenerateAnswerChoices(dspy.Signature):
    """Generate answer choices in JSON format that include the correct answer and plausible distractors for the specified question."""
    question = dspy.InputField()
    correct_answer = dspy.InputField()
    number_of_choices = dspy.InputField()
    answer_choices = dspy.OutputField(desc='JSON key-value pairs')

class QuizAnswerGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_choices = dspy.ChainOfThought(GenerateAnswerChoices)

    def forward(self, question, answer):
        choices = self.generate_choices(question=question, correct_answer=answer, number_of_choices=number_of_choices).answer_choices
        return dspy.Prediction(choices = choices)

number_of_choices = '4'
quiz_generator = QuizAnswerGenerator()


def format_checker(choice_string):
    try:
        choices = json.loads(choice_string)
        if isinstance(choices, dict) and all(isinstance(key, str) and isinstance(value, str) for key, value in choices.items()):
            return True
    except json.JSONDecodeError:
        return False

    return False

def is_correct_answer_included(correct_answer, generated_choices):
    try:
        choices_dict = json.loads(generated_choices)
        return correct_answer in choices_dict.values()
    except json.JSONDecodeError:
        return False

def is_plausibility_yes(assessment_answer):
    """Check if the first word of the assessment answer is 'yes'."""
    return assessment_answer.split()[0].lower() == 'yes'
    
class AssessQuizChoices(dspy.Signature):
    """Assess the quality of quiz answer choices along specified dimensions."""
    
    question = dspy.InputField()
    answer_choices = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")
    
def format_valid_metric(gold, pred, trace=None):
    generated_choices = pred.choices
    format_valid = format_checker(generated_choices)
    score = format_valid
    return score

def is_correct_metric(gold, pred, trace=None):
    correct_answer, generated_choices = gold.answer, pred.choices
    correct_included = is_correct_answer_included(correct_answer, generated_choices)
    score = correct_included
    return score

def plausibility_metric(gold, pred, trace=None):
    question, generated_choices = gold.question, pred.choices
    plausibility_question = "Are the distractors in the answer choices plausible and not easily identifiable as incorrect?"
    plausibility_assessment = dspy.Predict(AssessQuizChoices)(question=question, answer_choices=generated_choices, assessment_question=plausibility_question)
    plausibility_result = plausibility_assessment.assessment_answer.split()[0].lower() == 'yes'
    score = plausibility_result
    return score

def overall_metric(gold, pred, trace=None):
    question, correct_answer, generated_choices = gold.question, gold.answer, pred.choices
    format_valid = format_checker(generated_choices)
    correct_included = is_correct_answer_included(correct_answer, generated_choices)
    plausibility_question = "Are the distractors in the answer choices plausible and not easily identifiable as incorrect?"
    plausibility_assessment = dspy.Predict(AssessQuizChoices)(question=question, answer_choices=generated_choices, assessment_question=plausibility_question)
    plausibility_result = plausibility_assessment.assessment_answer.split()[0].lower() == 'yes'
    score = (format_valid + correct_included + plausibility_result) / 3.0 if correct_included and format_valid else 0
    return score


"""
metrics = [format_valid_metric, is_correct_metric, plausibility_metric, overall_metric]

for metric in metrics:
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(quiz_generator)
    
example = devset[67]
quiz_choices = quiz_generator(question=example.question, answer = example.answer)
print(f'Generated Quiz Choices: ', quiz_choices.choices)

for metric in metrics:
    evaluate = Evaluate(metric=metric, devset=devset[67:68], num_threads=1, display_progress=True, display_table=5)
    evaluate(quiz_generator)
    
"""
    
class QuizAnswerGeneratorWithAssertions(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_choices = dspy.ChainOfThought(GenerateAnswerChoices)

    def forward(self, question, answer):
        choice_string = self.generate_choices(question=question, correct_answer=answer, number_of_choices=number_of_choices).answer_choices
        dspy.Suggest(format_checker(choice_string), "The format of the answer choices should be in JSON format. Please revise accordingly.", target_module=GenerateAnswerChoices)
        dspy.Suggest(is_correct_answer_included(answer, choice_string), "The answer choices do not include the correct answer to the question. Please revise accordingly.", target_module=GenerateAnswerChoices)
        plausibility_question = "Are the distractors in the answer choices plausible and not easily identifiable as incorrect?"
        plausibility_assessment = dspy.Predict(AssessQuizChoices)(question=question, answer_choices=choice_string, assessment_question=plausibility_question)
        dspy.Suggest(is_plausibility_yes(plausibility_assessment.assessment_answer), "The answer choices are not plausible distractors or are too easily identifiable as incorrect. Please revise to provide more challenging and plausible distractors.", target_module=GenerateAnswerChoices)
        return dspy.Prediction(choices = choice_string)

number_of_choices = '4'

quiz_generator_with_assertions = assert_transform_module(QuizAnswerGeneratorWithAssertions().map_named_predictors(Retry), backtrack_handler) 


metrics = [format_valid_metric, is_correct_metric, plausibility_metric, overall_metric]

"""
for metric in metrics:
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(quiz_generator_with_assertions)
    
example = devset[67]
quiz_choices = quiz_generator_with_assertions(question=example.question, answer = example.answer)
print(f'Generated Quiz Choices: ', quiz_choices.choices)

for metric in metrics:
    evaluate = Evaluate(metric=metric, devset=devset[67:68], num_threads=1, display_progress=True, display_table=30)
    evaluate(quiz_generator_with_assertions)

    
teleprompter = BootstrapFewShotWithRandomSearch(metric = overall_metric, max_bootstrapped_demos=2, num_candidate_programs=6)
compiled_quiz_generator = teleprompter.compile(student = quiz_generator, teacher = quiz_generator, trainset=trainset, valset=devset[:100])

for metric in metrics:
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(compiled_quiz_generator)
    
    
teleprompter = BootstrapFewShotWithRandomSearch(metric = overall_metric, max_bootstrapped_demos=2, num_candidate_programs=6)
compiled_with_assertions_quiz_generator = teleprompter.compile(student=quiz_generator, teacher = quiz_generator_with_assertions, trainset=trainset, valset=devset[:100])

for metric in metrics:
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(compiled_with_assertions_quiz_generator)
    
"""
    
teleprompter = BootstrapFewShotWithRandomSearch(metric = overall_metric, max_bootstrapped_demos=2, num_candidate_programs=6)
compiled_quiz_generator_with_assertions = teleprompter.compile(student=quiz_generator_with_assertions, teacher = quiz_generator_with_assertions, trainset=trainset, valset=devset[:100])

for metric in metrics:
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=1, display_progress=True, display_table=5)
    evaluate(compiled_quiz_generator_with_assertions)
import os
from dotenv import load_dotenv
from agents import Chainer, Blogger, Router
from evals import evaluate_data_chain, evaluate_blog_post, evaluate_router

def setup():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    return api_key

api_key = setup()
def test_chainer():
    data_processing_steps = [
        """Extract only the numerical values and their associated metrics from the text.
        Format each as 'value: metric' on a new line.
        Example format:
        92: customer satisfaction
        45%: revenue growth""",
        
        """Convert all numerical values to percentages where possible.
        If not a percentage or points, convert to decimal (e.g., 92 points -> 92%).
        Keep one number per line.
        Example format:
        92%: customer satisfaction
        45%: revenue growth
        23.00: customer acquisition cost""",
        
        """Sort all lines in descending order by numerical value.
        Keep the format 'value: metric' on each line.
        Example:
        92%: customer satisfaction
        87%: employee satisfaction""",
        
        """Format the sorted data as a markdown table with columns, without tags or enclosing quotes:
        | Metric | Value |
        |:--|--:|
        | Customer Satisfaction | 92% |"""
    ]

    system_prompt = "You are a helpful assistant specializing in marketing analysis."

    report = """
    Q3 Performance Summary:
    Our customer satisfaction score rose to 92 points this quarter.
    Revenue grew by 45% compared to last year.
    Market share is now at 23% in our primary market.
    Customer churn decreased to 5% from 8%.
    New user acquisition cost is $43 per user.
    Product adoption rate increased to 78%.
    Employee satisfaction is at 87 points.
    Operating margin improved to 34%.
    """

    chainer = Chainer(system_prompt, data_processing_steps, report, "gpt-4o-mini", api_key)
    output = chainer.chain(verbose=False)
    compare = evaluate_data_chain(report, output)
    assert compare["all_metrics_captured"] == 8
    # assert compare["correct_formatting"] == True
    assert compare["sorting_accuracy"] == True

def test_blogger():
    topic = "The Future of Artificial Intelligence"
    target_audience = "tech enthusiasts"
    target_word_count = 800
    blogger = Blogger(topic, target_audience, target_word_count, "gpt-4o-mini", api_key)
    post = blogger.chain()
    compare = evaluate_blog_post(post, topic, target_audience, target_word_count)
    assert compare["word_count_match"] == True
    assert compare["has_introduction"] == True
    assert compare["has_conclusion"] == True
    assert compare["topic_relevance"] >= 0.05
    assert compare["structure_score"] >= 0.2

def test_router():
    router = Router("gpt-4o-mini", api_key)
    test_tickets = [
    {
        "ticket": """Ticket 1:
        Subject: Can't log in to my account
        Message: I've been trying to log in for the past hour but keep getting an "invalid password" error. 
        I know my password is correct because I use a password manager.
        Thanks,
        John""",
        "expected": "account"
    },
    {
        "ticket": """Ticket 2:
        Subject: Unexpected charge on my card
        Message: Hello, I just noticed a charge of $49.99 on my credit card from your company, but I thought
        I was on the $29.99 plan. Can you explain this charge and adjust it if it's a mistake?
        Thanks,
        Sarah""",
        "expected": "billing"
    },
    {
        "ticket": """Ticket 3:
        Subject: How to export data?
        Message: I need to export all my project data to Excel. I've looked through the docs but can't
        figure out how to do a bulk export. Is this possible? If so, could you walk me through the steps?
        Best regards,
        Mike""",
        "expected": "technical"
    },
    {
        "ticket": """Ticket 4:
        Subject: Feature request - calendar integration
        Message: I love your product but really wish it could integrate with Google Calendar. 
        Is this something you're planning to add in the future?
        Regards,
        Lisa""",
        "expected": "product"
    }
]

    # Run evaluation
    router_results = evaluate_router(router, test_tickets)
    print(f"Router accuracy: {router_results['accuracy']*100:.1f}%")
    if router_results["misclassifications"]:
        print("\nMisclassifications:")
    for misclass in router_results["misclassifications"]:
        print(f"Expected: {misclass['expected']}, Predicted: {misclass['predicted']}")
        print(f"Ticket: {misclass['ticket']}")
        print(f"Response: {misclass['response']}\n")
    assert router_results["accuracy"] == 1.0

if __name__ == "__main__":
    test_chainer()
    test_blogger()
    test_router()
    print("All tests passed!")
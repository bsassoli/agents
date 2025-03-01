from openai import OpenAI

api_key = "sk-proj-1234567890"
client = OpenAI(api_key=api_key)

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
    45%: revenue growth""",
    
    """Sort all lines in descending order by numerical value.
    Keep the format 'value: metric' on each line.
    Example:
    92%: customer satisfaction
    87%: employee satisfaction""",
    
    """Format the sorted data as a markdown table with columns:
    | Metric | Value |
    |:--|--:|
    | Customer Satisfaction | 92% |"""
]
def chain_llm(system_prompt, steps, input_, model):
    chain = [system_prompt]
    for ix, step in enumerate(steps):
        print(f"STEP: {ix}")
        new_step = {
            "role": "user",
            "content":
            [{
                "type": "text",
                "text":f"{step}\nInput: {input_}\n"
            }]
        }        
        chain.append(new_step)
        print(chain)
        response = client.chat.completions.create(
            model=model,
            messages=chain,
            max_completion_tokens=3000)
        response_content = response.choices[0].message.content
        print(response_content)
        input_= response_content
    return response_content

system_prompt = {
        "role": "developer",
        "content": [{
            "type": "text",
            "text": "You are a helpful assistant specializing in marketing analysis."
    }]}

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

out = chain_llm(system_prompt, data_processing_steps, report, "gpt-4o-mini")
print(out)
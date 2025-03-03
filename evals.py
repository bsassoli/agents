def evaluate_data_chain(original_text, processed_output):
    """Evaluate the data processing chain results"""
    metrics = {
        "all_metrics_captured": 0,
        "correct_formatting": False,
        "sorting_accuracy": False
    }
    
    # Check if all metrics from original text were captured
    original_metrics = []
    for line in original_text.split('\n'):
        if '%' in line or 'points' in line or '$' in line:
            metrics["all_metrics_captured"] += 1
    
    # Check if final output follows markdown table format
    if "| Metric | Value |" in processed_output and "|:--|--:|" in processed_output:
        metrics["correct_formatting"] = True
    
    # Check if sorting was applied correctly (should be descending)
    values = []
    for line in processed_output.split('\n'):
        if '|' in line and '%' in line:
            try:
                value = float(line.split('|')[2].strip().replace('%', ''))
                values.append(value)
            except:
                pass
    
    metrics["sorting_accuracy"] = all(values[i] >= values[i+1] for i in range(len(values)-1))
    
    return metrics

def evaluate_blog_post(post, topic, target_audience, target_word_count):
    """Evaluate the generated blog post"""
    words = len(post.split())
    
    metrics = {
        "word_count": words,
        "word_count_match": abs(words - target_word_count) / target_word_count <= 0.2,
        "has_introduction": "Introduction" in post or "## Introduction" in post,
        "has_conclusion": "Conclusion" in post or "## Conclusion" in post,
        "topic_relevance": sum(post.lower().count(word.lower()) for word in topic.split()) / words
    }
    
    # Count number of headings (structure)
    heading_count = 0
    for line in post.split('\n'):
        if line.strip().startswith('#'):
            heading_count += 1
    
    metrics["structure_score"] = min(heading_count / 5, 1.0)  # Normalize to 0-1
    
    return metrics

def evaluate_router(router, test_cases):
    """Evaluate router performance on test cases"""
    results = {
        "accuracy": 0,
        "confidence": 0,
        "misclassifications": []
    }
    
    correct = 0
    for case in test_cases:
        router.input_data = case["ticket"]
        result = router.route()
        
        # Extract the department from the result
        # This assumes the format "The most appropriate option is 'department'"
        predicted_dept = None
        for choice in router.choices:
            if f"'{choice}'" in result or f'"{choice}"' in result:
                predicted_dept = choice
                break
        
        if predicted_dept == case["expected"]:
            correct += 1
        else:
            results["misclassifications"].append({
                "ticket": case["ticket"],
                "expected": case["expected"],
                "predicted": predicted_dept,
                "response": result
            })
    
    results["accuracy"] = correct / len(test_cases)
    
    return results
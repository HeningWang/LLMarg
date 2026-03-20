# test_listener_experiment.py 
# 基本函数功能测试

from listener_experiment import *

def test_parse_observation():
    """测试observation解析"""
    test_cases = [
        ("[3, 3, 3, 3, 3]", [3, 3, 3, 3, 3]),
        ("[9, 9, 9, 9, 9]", [9, 9, 9, 9, 9]),
    ]
    
    for input_str, expected in test_cases:
        result = parse_observation(input_str)
        assert result == expected, f"输入: {input_str}, 期望: {expected}, 得到: {result}"
    
    print("✓ parse_observation 测试通过")

def test_format_exam_table():
    """测试考试表格格式化"""
    correct_counts = [3, 3, 3, 3, 3]
    result = format_exam_table(correct_counts, 12)
    
    print("格式化结果:")
    print(result)
    
    assert "学生1: 答对3/12题 (25%)" in result
    print("✓ format_exam_table 测试通过")

def test_build_sentence():
    """测试句子构建"""
    result = build_sentence("all", "some", "right")
    expected = "all of the students got some of the answers right"
    assert result == expected, f"得到: {result}"
    
    print("✓ build_sentence 测试通过")

def test_parse_response():
    """测试响应解析"""
    test_cases = [
        ("high", "high"),
        ("low", "low"),
        ("info", "info"),
    ]
    
    for input_text, expected in test_cases:
        result = parse_response(input_text)
        assert result == expected, f"输入: '{input_text}', 期望: {expected}"
    
    print("✓ parse_response 测试通过")

if __name__ == "__main__":
    print("运行单元测试...\n")
    test_parse_observation()
    test_format_exam_table()
    test_build_sentence()
    test_parse_response()
    print("\n所有测试通过！")
import fitz  # PyMuPDF
import re
import os
from datetime import datetime

# 创建输出文件夹
OUTPUT_DIR = "output_analysis"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def extract_text_from_pdf(pdf_path):
    """从PDF提取文本"""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return ""

def extract_financial_indicators(text):
    """提取关键财务指标"""
    indicators = {
        "总资产": r"总资产[：:\s]*([0-9,\.\-]+)",
        "总负债": r"总负债[：:\s]*([0-9,\.\-]+)",
        "净利润": r"(?:净利润|归属于母公司所有者的净利润)[：:\s]*([0-9,\.\-]+)"
    }
    results = {}
    for key, pattern in indicators.items():
        match = re.search(pattern, text)
        results[key] = match.group(1) if match else "未找到"
    return results

def analyze_risk_keywords(text):
    """统计风险相关关键词频率"""
    keywords = ["风险", "下滑", "减值"]
    return {kw: text.count(kw) for kw in keywords}

def generate_markdown_report(pdf_path, text, indicators, risks):
    """生成Markdown格式的分析报告"""
    # 获取PDF文件名（不含路径和扩展名）
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # 生成报告内容
    report = f"""# PDF文档分析报告

## 1. 文档信息
- **文件名**: {os.path.basename(pdf_path)}
- **分析时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 2. 关键财务指标
| 指标 | 数值 |
|------|------|
"""
    
    # 添加财务指标表格
    for key, value in indicators.items():
        report += f"| {key} | {value} |\n"
    
    # 添加风险分析部分
    report += "\n## 3. 风险分析\n"
    report += "| 关键词 | 出现次数 |\n"
    report += "|--------|----------|\n"
    for word, count in risks.items():
        report += f"| {word} | {count} |\n"
    
    # 添加文本统计信息
    total_chars = len(text)
    total_lines = text.count('\n') + 1
    
    report += f"""
## 4. 文本统计
- **总字符数**: {total_chars}
- **总行数**: {total_lines}

## 5. 分析说明
- 本报告由自动分析工具生成
- 使用工具: PyMuPDF
- 分析内容包括：财务指标提取、风险关键词统计
- 如有疑问请核对原始PDF文件
"""
    
    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, f"{pdf_name}_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report_path

def main():
    pdf_path = "C:\\Users\\karen\\Downloads\\Data-Copilot-try1\\text_data\\1223413786.pdf"
    
    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        print("⚠️ 无效路径或文件不是PDF")
        return

    print(f"\n📄 正在分析：{pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("❌ 文本提取失败，无法继续分析")
        return

    # 提取财务指标
    print("📊 正在提取财务指标...")
    indicators = extract_financial_indicators(text)

    # 分析风险词频
    print("⚠️ 正在分析风险关键词...")
    risks = analyze_risk_keywords(text)

    # 生成Markdown报告
    print("📝 正在生成分析报告...")
    report_path = generate_markdown_report(pdf_path, text, indicators, risks)
    
    print("\n✅ 分析完成！")
    print(f"📊 分析报告已保存到：{report_path}")
    print(f"📁 可以在 '{OUTPUT_DIR}' 文件夹中查看完整报告。")

if __name__ == "__main__":
    main()

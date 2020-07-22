from ai_harness.xml2object import parse

xml_text = '''
<doc id="0"><summary>修改后的立法法全文公布</summary>
<short_text>
新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。
</short_text>
</doc>
'''


class Test_Data_Processor():
    def test_xml_to_json(self):
        obj = parse(xml_text.replace('\n', ''))
        print(obj.doc.summary.cdata, obj.doc.short_text.cdata)

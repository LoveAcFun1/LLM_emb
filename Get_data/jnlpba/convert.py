import json
labels = {"cell_line":"cell line", "protein":"protein", "RNA":"RNA", "DNA":"DNA", "cell_type":"cell type"}
out_path = "/home/lqb/AT_llama/Get_data/jnlpba/dev.json"
f1 = open(out_path, "w")
def convert_ner_to_json(input_file, output_file):  
    # 初始化input和output字符串  
    input_str = ""  
    output_str = ""  
    current_entity = ""  
    in_entity = False 
    data = [] 
  
    with open(input_file, 'r', encoding='utf-8') as file:  
        for line in file:  
            # 跳过空行和-DOCSTART-行  
            if line.startswith("-DOCSTART-"):  
                continue 
            if line.strip() == "":
                if  input_str == "":
                    continue
                output_str = output_str.strip()
                if output_str == "":
                    output_str = " None "  
                json_data = {  
                    "input": input_str.strip(),  
                    "output": output_str[:-1]  
                }  
                data.append(json_data)
                input_str = ""  
                output_str = ""  
                current_entity = ""
                continue
  
            # 分割每行的内容  
            word, _, tag1, tag2 = line.strip().split('\t')  
            input_str += word + " " 
            # 如果遇到B-开头的实体标签，开始记录实体  
            if tag2.startswith("B-"):
                if in_entity:
                    current_entity = labels[tag2[2:]]  
                    in_entity = True
                    output_str += " ;" + current_entity + ": " + word 
                else:  
                    current_entity = labels[tag2[2:]]  
                    in_entity = True  
                    output_str += " " + current_entity + ": " + word 
  
            # 如果遇到I-开头的实体标签，继续添加到当前实体  
            elif tag2.startswith("I-") and in_entity:  
                output_str += " " + word  
  
            # 如果遇到非实体标签或实体结束，则记录input字符串，并可能结束当前实体  
            else:   
                if in_entity:  
                    output_str += ";"  
                    in_entity = False  
  
        # 处理最后一个实体  
        if in_entity:  
            output_str += " " + word  
            in_entity = False  
        json_data = {  
                    "input": input_str.strip(),  
                    "output": output_str  
                }  
        data.append(json_data)
        for di in data:
            f1.write(json.dumps(di))
            f1.write('\n')
    # 去除output_str末尾的空格  
    
  
    # 构造JSON对象  
    # 将JSON对象写入文件  
    # with open(output_file, 'w', encoding='utf-8') as file:  
    #     json.dump(json_data, file, ensure_ascii=False, indent=4)  
  
# 调用函数，将ner_input.txt转换为json格式的文件  
convert_ner_to_json('/home/lqb/AT_llama/Get_data/jnlpba/dev.txt', 'output.json')
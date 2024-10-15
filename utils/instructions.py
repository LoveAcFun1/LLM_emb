import random
class Instruct:
    def __init__(self, mode) -> None:
        self.mode = mode
        
    def get_ins(self):
        instructs = ["Please list all entity words in the text that fit the category.Output format is \"type1: word1; type2: word2\". \n.  If the above category entities do not exist in the sentence, None is output.",
                     "Please find all the entity words associated with the category in the given text.Output format is \"type1: word1; type2: word2\". \n If the above category entities do not exist in the sentence, None is output.",
                     "Please tell me all the entity words in the text that belong to a given category.Output format is \"type1: word1; type2: word2\". \n If the gene-related phrase does not exist in the sentence, None is output."]
        if self.mode == "random":
            return random.choice(instructs)
        else:
            return instructs[self.mode]
    
    def bc2gm(self):
        instructs = ["Your task is to identify and label gene-related Named Entities within the text. Marking the words of a gene-related phrase as form \"type1: word1; type2: word2\". If the above category entities do not exist in the sentence, None is output.",
                     "In the provided text, your objective is to recognize and tag gene-related Named Entities. Labeling the words of a gene-related phrase as form \"type1: word1; type2: word2\". If the above category entities do not exist in the sentence, None is output.",
                     "In the provided text, your goal is to identify and label gene-related Named Entities. For gene-related phrases mark them as \"type1: word1; type2: word2\". If the gene-related phrase does not exist in the sentence, None is output."]
        if self.mode == "random":
            return random.choice(instructs)
        else:
            return instructs[self.mode]
    def AnatEM(self):
        instructs = ["Your task is to identify and label anatomy-related Named Entities within the text. Marking the words of an anatomy-related phrase as form \"type1: word1; type2: word2\". If the above category entities do not exist in the sentence, None is output.",
             "In the provided text, your objective is to recognize and tag anatomy-related Named Entities. Labeling the words of an anatomy-related phrase as form \"type1: word1; type2: word2\". If the above category entities do not exist in the sentence, None is output.",
             "In the provided text, your goal is to identify and label anatomy-related Named Entities. For anatomy-related phrases mark them as \"type1: word1; type2: word2\". If the anatomy-related phrase does not exist in the sentence, None is output."]
        if self.mode == "random":
            return random.choice(instructs)
        else:
            return instructs[self.mode]
    def bc4chemd(self):
        instructs = ["Your task is to identify and label chemical-related Named Entities within the text. Mark the words of chemical-related phrases with format \"type1: word1; type2: word2\". If no chemical-related entities are found in the sentence, output None.",
             "In the provided text, your objective is to recognize and tag chemical-related Named Entities. Label the words of chemical-related phrases with format \"type1: word1; type2: word2\". If no chemical-related entities are found in the sentence, output None.",
             "In the provided text, your goal is to identify and label chemical-related Named Entities. For chemical-related phrases, mark them with format \"type1: word1; type2: word2\". If no chemical-related entities are found in the sentence, output None."]
        if self.mode == "random":
            return random.choice(instructs)
        else:
            return instructs[self.mode]
    # def bc5cdr(self):
    #     instructs = []
    #     if self.mode == "random":
    #         return random.choice(instructs)
    #     else:
    #         return instructs[self.mode]
    # def GENIA_NER(self):
    #     if self.mode == "random":
    #         return random.choice(instructs)
    #     else:
    #         return instructs[self.mode]
    # def JNLPBA(self):
    #     if self.mode == "random":
    #         return random.choice(instructs)
    #     else:
    #         return instructs[self.mode]
    # def ncbi(self):
    #     if self.mode == "random":
    #         return random.choice(instructs)
    #     else:
    #         return instructs[self.mode]
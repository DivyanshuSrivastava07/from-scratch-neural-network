s = "hello"
class CharacterTokenizer:
    def __init__(self,corpus):
        self.stoi: dict = {}
        for ch in corpus:
            if ch not in self.stoi.keys():
                self.stoi[ch] = len(self.stoi)
            
        self.itos = {v: k for k, v in self.stoi.items()}
    def encode(self,text:str) -> list[int]:
        l = []    
        for char in text:
            l.append(self.stoi[char])
        return l

    def decode(self,tokens:list[int]) -> str:
        text = ""

        for i in tokens:
            text += self.itos[i]
        return text
    @property
    def vocab_size(self):
        return len(self.itos)
ct = CharacterTokenizer("hellowret")

l = ct.encode(s)
print(ct.vocab_size)

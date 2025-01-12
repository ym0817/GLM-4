import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyGLM4():
    '''
    GLM4 model
    '''
    def __init__(
            self,
            model_path: str = "./glm-4-9b-chat-int4",
            max_new_tokens: int = 512,
            do_sample: bool = True,
            top_k: int = 5,
            vision: bool = False,
            system_prompt = "你是一个人工智能助手，请认真回答下面的问题",
            device: str = "cuda" ,
            multi_user_list: list = [],
            multi_user_system_prompt: dict = {}
            ) -> None:
        
        # Build model and tokenizer
        print("Loading model and tokenizer...")
        tokenizer_path = "THUDM/glm-4-9b-chat" if not vision else "THUDM/glm-4v-9b"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print("Building Model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            load_in_4bit=True,
            torch_dtype=torch.float16
        ).eval()
        
        

        # Set generation kwargs
        self.device = device
        self.gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "top_k": top_k}
        self.system_prompt = system_prompt

        # Multi user setting. For WeChat listening, message from different friends should be stored separately
        self.multi_user_flag = False if len(multi_user_list) == 0 else True
        self.multi_user_list = multi_user_list
        self.list_memory = {}

        if self.multi_user_flag:
            for user_id in multi_user_list:
                self.list_memory[user_id] = []
                if user_id in multi_user_system_prompt:
                    self.list_memory[user_id].append({"role": "system", "content": multi_user_system_prompt[user_id]})
                    print("assign custom system prompt for user:", user_id)
                else:
                    self.list_memory[user_id].append({"role": "system", "content": system_prompt})
        else:
            self.list_memory = []
            self.list_memory.append({"role": "system", "content": system_prompt})


    def get_response(self, query, user_id = None, img = None):
        # For WeChat listening, message from different friends should be stored separately
        if self.multi_user_flag:
            if img is not None:
                self.list_memory[user_id].append({"role": "user", "image": img, "content": query})
            else: 
                self.list_memory[user_id].append({"role": "user", "content": query})
            inputs = self.tokenizer.apply_chat_template(self.list_memory[user_id],
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    return_dict=True)

            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # add response to inputs_querys
                self.list_memory[user_id].append({"role": "assistant", "content": response})
                return response
        # For single user
        else:
            if img is not None:
                self.list_memory.append({"role": "user", "image": img, "content": query})
            else:
                self.list_memory.append({"role": "user", "content": query})
            inputs = self.tokenizer.apply_chat_template(self.list_memory,
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    return_dict=True)

            inputs = inputs.to(self.device)

            with torch.no_grad():
                if img is not None:
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        position_ids=inputs['position_ids'],
                        images=inputs['images'],
                        **self.gen_kwargs
                    )
                else:
                    outputs = self.model.generate(**inputs, **self.gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # add response to inputs_querys
                self.list_memory.append({"role": "assistant", "content": response})
                return response
    # def add_image(self, image, user_id = None):
    #     if self.multi_user_flag:
    #         self.list_memory[user_id].append({"image": image})
    #     else:
    #         self.list_memory.append({"image": image})

    # In case the chat memory accumulates too much, release the memory when necessary
    def release_chat_memory(self, user_id):
        self.list_memory[user_id] = []
        self.list_memory[user_id].append({"role": "system", "content": self.system_prompt})

if __name__ == "__main__":

    device = "cpu"

    # tokenizer = AutoTokenizer.from_pretrained("./glm-4-9b-chat", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        # "THUDM/glm-4v-9b",
        "THUDM/glm-4-9b-chat",
        # cache_dir="D:\\LLM",
        # torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        load_in_4bit=True
    ).eval()
    # model.save_pretrained("glm-4v-9b-chat-int4")
    model.save_pretrained("glm-4-9b-chat-int4")
    # tokenizer.save_pretrained("glm-4-9b-chat-int4")
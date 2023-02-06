# File to do some run some quick code snippets
import main 

# import Stable_Diffusion.txt2img_gpu as onnx


# prompt = "guy sitting in front of computer"



# model_id = "stable_diffusion_onnx"    
# pipe = onnx.onnxPipeline()


# img_path = 'test.jpg'
# neg_prompt = ""
# image = pipe(prompt, negative_prompt=neg_prompt).images[0]  

# image.save(img_path)


new_dir = main.get_new_file_name("created_images/")
print(new_dir)



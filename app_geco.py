# Code borrowed heavily from OpenLRM: https://huggingface.co/spaces/zxhezexin/OpenLRM/blob/main/app.py
import os
from PIL import Image
import numpy as np
import gradio as gr
from infer import Inferrer
import tyro
import rembg
from src.options import AllConfigs, Options, config_defaults
from src.utils.infer_util import remove_background, resize_foreground, images_to_video

def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")

def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir

def preprocess_fn(input_image: Image, do_remove_background, ratio=0.85):
    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, ratio)
    return input_image


def demo(inferrer):

    def core_fn(input_image, working_dir):
        dump_video_path = os.path.join(working_dir.name, "video.mp4")
        dump_mesh_path = os.path.join(working_dir.name, "mesh.obj")
        cond = np.array(Image.open(input_image))
        mask = cond[..., 3:4] / 255
        cond_input = cond[..., :3] * mask + (1 - mask) * 255
        cond = cond[..., :3] * mask + (1 - mask) * 255
        inferrer.infer(cond, working_dir.name, use_cond=True, demo=True)
        return dump_mesh_path

    def example_fn(image: np.ndarray):
        from gradio.utils import get_cache_folder
        working_dir = get_cache_folder()
        image = preprocess_fn(
            image_in=image,
            do_remove_background=True
        )
        mesh, video = core_fn(
            image=image,
            working_dir=working_dir,
        )
        return image, mesh, video


    _TITLE = '''GECO: Generative Image-to-3D within a SECOnd'''

    _DESCRIPTION = '''
        <strong>Image-to-3D mesh in 1 seconds!</strong>
    '''

    with gr.Blocks(analytics_enabled=False) as demo:

        # HEADERS
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown('# ' + _TITLE)
        with gr.Row():
            gr.Markdown(_DESCRIPTION)

        # DISPLAY
        with gr.Row(variant='panel'): # left

            with gr.Column(variant='panel', scale=1):
                with gr.Row():
                    input_image = gr.Image(label="Input Image", image_mode="RGBA", width="auto", sources="upload", type="pil", elem_id="content_image")
                    processed_image = gr.Image(label="Processed Image", image_mode="RGBA", type="filepath", elem_id="processed_image", width="auto", interactive=False)

                gr.Markdown(
                    """
                    <strong>Best Practice</strong>:
                        Centered objects in reasonable sizes.
                    """
                )
                with gr.Row():
                    checkbox_rembg = gr.Checkbox(True, label='Remove background')
                    # checkbox_recenter = gr.Checkbox(True, label='Recenter the object')
                    center_ratio = gr.Slider(0.5, 0.85, value=0.85, step=0.01, label="Padding ratio")
                with gr.Row():
                    submit = gr.Button('Generate', elem_id="generate", variant='primary')

            with gr.Column(variant='panel', scale=1):
                    # output_video = gr.Video(label="Rendered Video", format="mp4", width="auto", autoplay=True, interactive=False)
                with gr.Row():
                    with gr.Tab("OBJ"):
                        output_model_obj = gr.Model3D(
                            label="Output Model (OBJ Format)",
                            #width=768,
                            interactive=False,
                        )
                        gr.Markdown("Note: Downloaded .obj model will be flipped. Export .glb instead or manually flip it before usage.")
                    with gr.Tab("GLB"):
                        output_model_glb = gr.Model3D(
                            label="Output Model (GLB Format)",
                            #width=768,
                            interactive=False,
                        )
                        gr.Markdown("Note: The model shown here has a darker appearance. Download to get correct results.")

        # # EXAMPLES
        with gr.Row():
            examples = [
                ['examples/mushroom_teapot.jpg'],
                ['examples/fox.jpg'],
                ['examples/dora_wong.png'],
                ['examples/skating_shoe.jpg'],
                ['examples/teasure_chest.jpg']
            ]
            gr.Examples(
                examples=examples,
                inputs=[input_image], 
                outputs=[processed_image, output_model_obj],
                fn=example_fn,
                cache_examples=bool(os.getenv('SPACE_ID')),
                examples_per_page=20,
            )

        working_dir = gr.State()
        gaussians = gr.State()
        submit.click(
            fn=assert_input_image,
            inputs=[input_image],
            queue=False,
        ).success(
            fn=prepare_working_dir,
            outputs=[working_dir],
            queue=False,
        ).success(
            fn=preprocess_fn,
            inputs=[input_image, checkbox_rembg, center_ratio],
            outputs=[processed_image],
        ).success(
            fn=core_fn,
            inputs=[processed_image, working_dir],
            outputs=[output_model_obj],
        )

        demo.queue()
        demo.launch(share=True)


def launch_gradio_app():

    os.environ.update({
        # "APP_ENABLED": "1",
        # "APP_MODEL_NAME": "zxhezexin/openlrm-mix-base-1.1",
        # "APP_INFER": "./configs/infer-gradio.yaml",
        # "APP_TYPE": "infer.lrm",
        # "NUMBA_THREADING_LAYER": 'omp',
        "OMP_NUM_THREADS": "1",
    })

    opt = config_defaults['big']
    opt.generate_mv = True
    opt.pipeline = 'zero123plus1step'
    opt.cond_t = 0
    opt.init_t = 950
    opt.model_type = 'LGM'
    opt.joint = True
    opt.bg = 1.0
    opt.fovy = 50
    opt.distance = 2.0
    opt.config_path = 'configs/instant-mesh-base.yaml'
    opt.zero123plus_v = 'v11'
    opt.use_cond = True
    opt.resume = './ckpts/model.safetensors'
    opt.resume_pkl = './ckpts/network-snapshot-005000.pkl'
    inferer = Inferrer(opt)
    demo(inferer)


if __name__ == '__main__':
    launch_gradio_app()

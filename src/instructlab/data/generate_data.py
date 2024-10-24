# Standard
import logging

# Third Party
from instructlab.sdg.generate_data import generate_data

from concurrent.futures import ProcessPoolExecutor

# pylint: disable=ungrouped-imports
from instructlab.sdg.utils import GenerateException
import openai
from instructlab import log

import os

# First Party
from instructlab.utils import HttpClientParams, http_client

logger = logging.getLogger(__name__)


def gen_data(
    serve_cfg,
    model_path,
    num_cpus,
    sdg_scale_factor,
    taxonomy_path,
    taxonomy_base,
    output_dir,
    quiet,
    endpoint_url,
    api_key,
    yaml_rules,
    chunk_word_count,
    server_ctx_size,
    http_client_params: HttpClientParams,
    model_family,
    pipeline,
    enable_serving_output,
    batch_size,
    gpus,
    checkpoint_dir,
    num_servers,
):
    """Generates synthetic data to enhance your example data"""
    backend_instance = None

    if endpoint_url:
        api_base = endpoint_url
    else:
        # First Party
        from instructlab.model.backends import backends
        from instructlab.model.backends.llama_cpp import Server as llama_cpp_server

        backend_instances = []
        base_port = 8000
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                prepare_for_and_generate_data,
                pipeline,
                serve_cfg,
                http_client_params,
                model_family,
                model_path,
                num_cpus,
                sdg_scale_factor,
                taxonomy_path,
                taxonomy_base,
                output_dir,
                quiet,
                yaml_rules,
                chunk_word_count,
                server_ctx_size,
                batch_size,
                checkpoint_dir,
                thread,
                num_servers,
                )
                for thread in range(num_servers)
            ]
            for i, future in enumerate(futures):
                if future.running():
                    logger.debug(f"Thread {i} is running")
                elif future.done():
                    logger.debug(f"Thread {i} has completed")
                elif future.cancelled():
                    logger.debug(f"Thread {i} was canceled")
            for future in futures:
                ds = future.result()
                print(ds)
 

def prepare_for_and_generate_data(
        pipeline,
        serve_cfg,
        http_client_params,
        model_family,
        model_path,
        num_cpus,
        sdg_scale_factor,
        taxonomy_path,
        taxonomy_base,
        output_dir,
        quiet,
        yaml_rules,
        chunk_word_count,
        server_ctx_size,
        batch_size,
        checkpoint_dir,
        thread,
        total_threads,

):
    from instructlab.model.backends import backends
    from instructlab.model.backends.llama_cpp import Server as llama_cpp_server

    os.sched_setaffinity(0, {thread % os.cpu_count()})
    serve_cfg.host_port = f"127.0.0.1:{8000-thread}"
    log.add_file_handler_to_logger(logger, f"{8000-thread}.txt")
    backend_instance = backends.select_backend(cfg=serve_cfg, model_path=model_path, log_file=f"{8000-thread}.txt", num_threads=4)
    try:
        logger.info(
            f"Generating synthetic data using '{pipeline}' pipeline, '{model_path}' model, '{taxonomy_path}' taxonomy"
        )
        base = backend_instance.run_detached(
            http_client=http_client(http_client_params),
            background=False,
            foreground_allowed=True,
            max_startup_retries=1,
        )
        c = openai.OpenAI(
                base_url=base, api_key='no_api_key',  http_client=http_client(http_client_params)
        )
        # disable batching when running with the local llama.cpp server
        if isinstance(backend_instance, llama_cpp_server):
            if batch_size is not None:
                logger.warning(
                    "Disabling SDG batching - unsupported with llama.cpp serving"
                )
            batch_size = 0
        generate_data(
            client=c,
            model_family=model_family,
            model_name=model_path,
            num_cpus=num_cpus,
            num_instructions_to_generate=sdg_scale_factor,
            taxonomy=taxonomy_path,
            taxonomy_base=taxonomy_base,
            output_dir=output_dir,
            console_output=not quiet,
            yaml_rules=yaml_rules,
            chunk_word_count=chunk_word_count,
            server_ctx_size=server_ctx_size,
            pipeline=pipeline,
            batch_size=batch_size,
            checkpoint_dir=checkpoint_dir,
            thread=thread,
            total_threads=total_threads,
        )
    except KeyboardInterrupt as keyb:
        logger.info(f"Detected Keyboard Interrupt, shutting down all servers {keyb}")
        if backend_instance is not None:
            backend_instance.shutdown()
    except GenerateException as exc:
        raise ValueError(
            f"Generating dataset failed with the following error: {exc}"
        ) from exc
    finally:
        if backend_instance is not None:
            backend_instance.shutdown()

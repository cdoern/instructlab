# Standard
import logging

# Third Party
from instructlab.sdg.generate_data import generate_data

# pylint: disable=ungrouped-imports
from instructlab.sdg.utils import GenerateException
import openai
from instructlab import log


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
        for i in range(num_servers):
            serve_cfg.host_port = f"127.0.0.1:{base_port-i}"
            serve_cfg.llama_cpp.llm_family = "mixtral"
            try: 
                log.add_file_handler_to_logger(logger, f"{base_port-i}.txt")
                backend_instance = backends.select_backend(cfg=serve_cfg, model_path=model_path, log_file=f"{base_port-i}.txt", num_threads=2)

            except Exception as exc:
                print(exc)
                raise exc
            if (
                backend_instance.get_backend_type() is not backends.VLLM
                and gpus is not None
            ):
                logger.debug(
                    "Cannot specify '--gpus' with a llama-cpp backend, ignoring this flag."
                )
            if backend_instance.get_backend_type() is not backends.LLAMA_CPP and num_servers > 1:
                logger.debug(
                    "Cannot specify '--num-servers' with vLLM backend. Ignoring this flag"
                )
                num_servers = 1
            backend_instances.append(backend_instance)

        try:
            # Run the backend server
            api_base_list = []
            for backend_instance in backend_instances:
                try:
                    base = backend_instance.run_detached(
                        http_client=http_client(http_client_params),
                      #  background=not enable_serving_output,
                      #  foreground_allowed=True,
                      #  max_startup_retries=1,
                    )
                    api_base_list.append(base)
                except Exception as exc:
                    print(exc)
                    raise exc
        except Exception as exc:
            raise ValueError(f"Failed to start server: {exc}") from exc

        # disable batching when running with the local llama.cpp server
        if isinstance(backend_instances[0], llama_cpp_server):
            if batch_size is not None:
                logger.warning(
                    "Disabling SDG batching - unsupported with llama.cpp serving"
                )
            batch_size = 0
    try:
        logger.info(
            f"Generating synthetic data using '{pipeline}' pipeline, '{model_path}' model, '{taxonomy_path}' taxonomy"
        )
        generate_data(
            clients=api_base_list,
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
        )
    except KeyboardInterrupt as keyb:
        logger.info("Detected Keyboard Interrupt, shutting down all servers")
        if backend_instances is not None:
            for backend_instance in backend_instances:
                backend_instance.shutdown()
    except GenerateException as exc:
        raise ValueError(
            f"Generating dataset failed with the following error: {exc}"
        ) from exc
    finally:
        if backend_instances is not None:
            for backend_instance in backend_instances:
                backend_instance.shutdown()

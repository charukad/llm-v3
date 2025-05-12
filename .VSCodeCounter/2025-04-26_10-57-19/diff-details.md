# Diff Details

Date : 2025-04-26 10:57:19

Directory /Users/dasuncharuka/Desktop/new/math-llm-system/orchestration

Total : 65 files,  5592 codes, 556 comments, 1159 blanks, all 7307 lines

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [multimodal/\_\_init\_\_.py](/multimodal/__init__.py) | Python | -29 | 0 | -4 | -33 |
| [multimodal/agent/\_\_init\_\_.py](/multimodal/agent/__init__.py) | Python | -6 | 0 | -2 | -8 |
| [multimodal/agent/advanced\_ocr\_agent.py](/multimodal/agent/advanced_ocr_agent.py) | Python | -125 | -20 | -40 | -185 |
| [multimodal/agent/ocr\_agent.py](/multimodal/agent/ocr_agent.py) | Python | -104 | -11 | -28 | -143 |
| [multimodal/context/\_\_init\_\_.py](/multimodal/context/__init__.py) | Python | -3 | 0 | -1 | -4 |
| [multimodal/context/context\_manager.py](/multimodal/context/context_manager.py) | Python | -262 | -22 | -80 | -364 |
| [multimodal/context/reference\_resolver.py](/multimodal/context/reference_resolver.py) | Python | -203 | -39 | -68 | -310 |
| [multimodal/image\_processing/\_\_init\_\_.py](/multimodal/image_processing/__init__.py) | Python | -10 | 0 | -2 | -12 |
| [multimodal/image\_processing/coordinate\_detector.py](/multimodal/image_processing/coordinate_detector.py) | Python | -107 | -17 | -31 | -155 |
| [multimodal/image\_processing/diagram\_detector.py](/multimodal/image_processing/diagram_detector.py) | Python | -122 | -25 | -30 | -177 |
| [multimodal/image\_processing/format\_handler.py](/multimodal/image_processing/format_handler.py) | Python | -140 | -20 | -31 | -191 |
| [multimodal/image\_processing/preprocessor.py](/multimodal/image_processing/preprocessor.py) | Python | -221 | -30 | -68 | -319 |
| [multimodal/interaction/\_\_init\_\_.py](/multimodal/interaction/__init__.py) | Python | -3 | 0 | -1 | -4 |
| [multimodal/interaction/ambiguity\_handler.py](/multimodal/interaction/ambiguity_handler.py) | Python | -332 | -18 | -95 | -445 |
| [multimodal/interaction/feedback\_processor.py](/multimodal/interaction/feedback_processor.py) | Python | -160 | -12 | -49 | -221 |
| [multimodal/latex\_generator/\_\_init\_\_.py](/multimodal/latex_generator/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [multimodal/latex\_generator/latex\_generator.py](/multimodal/latex_generator/latex_generator.py) | Python | -140 | -4 | -37 | -181 |
| [multimodal/ocr/\_\_init\_\_.py](/multimodal/ocr/__init__.py) | Python | -10 | 0 | -2 | -12 |
| [multimodal/ocr/advanced\_symbol\_detector.py](/multimodal/ocr/advanced_symbol_detector.py) | Python | -118 | -26 | -42 | -186 |
| [multimodal/ocr/context\_analyzer.py](/multimodal/ocr/context_analyzer.py) | Python | -232 | -56 | -66 | -354 |
| [multimodal/ocr/performance\_optimizer.py](/multimodal/ocr/performance_optimizer.py) | Python | -163 | -25 | -50 | -238 |
| [multimodal/ocr/symbol\_detector.py](/multimodal/ocr/symbol_detector.py) | Python | -96 | -23 | -26 | -145 |
| [multimodal/structure/\_\_init\_\_.py](/multimodal/structure/__init__.py) | Python | -2 | 0 | -2 | -4 |
| [multimodal/structure/layout\_analyzer.py](/multimodal/structure/layout_analyzer.py) | Python | -175 | -26 | -49 | -250 |
| [multimodal/tests/\_\_init\_\_.py](/multimodal/tests/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [multimodal/tests/test\_context\_management.py](/multimodal/tests/test_context_management.py) | Python | -203 | -40 | -62 | -305 |
| [multimodal/tests/test\_ocr\_components.py](/multimodal/tests/test_ocr_components.py) | Python | -143 | -24 | -26 | -193 |
| [multimodal/tests/test\_unified\_pipeline.py](/multimodal/tests/test_unified_pipeline.py) | Python | -115 | -24 | -39 | -178 |
| [multimodal/unified\_pipeline/\_\_init\_\_.py](/multimodal/unified_pipeline/__init__.py) | Python | -4 | 0 | -1 | -5 |
| [multimodal/unified\_pipeline/content\_router.py](/multimodal/unified_pipeline/content_router.py) | Python | -105 | -7 | -27 | -139 |
| [multimodal/unified\_pipeline/input\_processor.py](/multimodal/unified_pipeline/input_processor.py) | Python | -249 | -24 | -47 | -320 |
| [orchestration/\_\_init\_\_.py](/orchestration/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [orchestration/agents/\_\_init\_\_.py](/orchestration/agents/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [orchestration/agents/agent\_info.py](/orchestration/agents/agent_info.py) | Python | 267 | 4 | 22 | 293 |
| [orchestration/agents/base\_agent.py](/orchestration/agents/base_agent.py) | Python | 435 | 49 | 103 | 587 |
| [orchestration/agents/capability\_advertisement.py](/orchestration/agents/capability_advertisement.py) | Python | 216 | 21 | 60 | 297 |
| [orchestration/agents/communication.py](/orchestration/agents/communication.py) | Python | 319 | 25 | 68 | 412 |
| [orchestration/agents/fault\_tolerance.py](/orchestration/agents/fault_tolerance.py) | Python | 307 | 58 | 91 | 456 |
| [orchestration/agents/load\_balancer.py](/orchestration/agents/load_balancer.py) | Python | 219 | 46 | 77 | 342 |
| [orchestration/agents/multimodal\_config.py](/orchestration/agents/multimodal_config.py) | Python | 106 | 3 | 18 | 127 |
| [orchestration/agents/registry.py](/orchestration/agents/registry.py) | Python | 212 | 19 | 50 | 281 |
| [orchestration/manager/\_\_init\_\_.py](/orchestration/manager/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [orchestration/manager/orchestration\_manager.py](/orchestration/manager/orchestration_manager.py) | Python | 404 | 79 | 121 | 604 |
| [orchestration/message\_bus/\_\_init\_\_.py](/orchestration/message_bus/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [orchestration/message\_bus/message\_formats.py](/orchestration/message_bus/message_formats.py) | Python | 197 | 5 | 36 | 238 |
| [orchestration/message\_bus/message\_handler.py](/orchestration/message_bus/message_handler.py) | Python | 204 | 21 | 52 | 277 |
| [orchestration/message\_bus/rabbitmq\_wrapper.py](/orchestration/message_bus/rabbitmq_wrapper.py) | Python | 267 | 29 | 58 | 354 |
| [orchestration/monitoring/\_\_init\_\_.py](/orchestration/monitoring/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [orchestration/monitoring/logger.py](/orchestration/monitoring/logger.py) | Python | 106 | 15 | 34 | 155 |
| [orchestration/monitoring/metrics.py](/orchestration/monitoring/metrics.py) | Python | 204 | 14 | 51 | 269 |
| [orchestration/monitoring/tracing.py](/orchestration/monitoring/tracing.py) | Python | 153 | 4 | 32 | 189 |
| [orchestration/tests/\_\_init\_\_.py](/orchestration/tests/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [orchestration/workflow/\_\_init\_\_.py](/orchestration/workflow/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [orchestration/workflow/end\_to\_end\_workflows.py](/orchestration/workflow/end_to_end_workflows.py) | Python | 259 | 37 | 73 | 369 |
| [orchestration/workflow/enhanced\_workflows.py](/orchestration/workflow/enhanced_workflows.py) | Python | 957 | 111 | 232 | 1,300 |
| [orchestration/workflow/error\_handler.py](/orchestration/workflow/error_handler.py) | Python | 233 | 21 | 54 | 308 |
| [orchestration/workflow/error\_recovery.py](/orchestration/workflow/error_recovery.py) | Python | 1,126 | 172 | 274 | 1,572 |
| [orchestration/workflow/integrated\_response\_workflow.py](/orchestration/workflow/integrated_response_workflow.py) | Python | 344 | 35 | 87 | 466 |
| [orchestration/workflow/multimodal\_workflows.py](/orchestration/workflow/multimodal_workflows.py) | Python | 383 | 22 | 60 | 465 |
| [orchestration/workflow/standard\_workflows.py](/orchestration/workflow/standard_workflows.py) | Python | 265 | 7 | 26 | 298 |
| [orchestration/workflow/visualization\_handler.py](/orchestration/workflow/visualization_handler.py) | Python | 352 | 64 | 94 | 510 |
| [orchestration/workflow/visualization\_workflows.py](/orchestration/workflow/visualization_workflows.py) | Python | 143 | 11 | 14 | 168 |
| [orchestration/workflow/workflow\_definition.py](/orchestration/workflow/workflow_definition.py) | Python | 185 | 7 | 37 | 229 |
| [orchestration/workflow/workflow\_engine.py](/orchestration/workflow/workflow_engine.py) | Python | 958 | 142 | 272 | 1,372 |
| [orchestration/workflow/workflow\_registry.py](/orchestration/workflow/workflow_registry.py) | Python | 353 | 28 | 64 | 445 |

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details
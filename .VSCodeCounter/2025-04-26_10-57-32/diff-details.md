# Diff Details

Date : 2025-04-26 10:57:32

Directory /Users/dasuncharuka/Desktop/new/math-llm-system/scripts

Total : 57 files,  -6099 codes, -637 comments, -1290 blanks, all -8026 lines

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [orchestration/\_\_init\_\_.py](/orchestration/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [orchestration/agents/\_\_init\_\_.py](/orchestration/agents/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [orchestration/agents/agent\_info.py](/orchestration/agents/agent_info.py) | Python | -267 | -4 | -22 | -293 |
| [orchestration/agents/base\_agent.py](/orchestration/agents/base_agent.py) | Python | -435 | -49 | -103 | -587 |
| [orchestration/agents/capability\_advertisement.py](/orchestration/agents/capability_advertisement.py) | Python | -216 | -21 | -60 | -297 |
| [orchestration/agents/communication.py](/orchestration/agents/communication.py) | Python | -319 | -25 | -68 | -412 |
| [orchestration/agents/fault\_tolerance.py](/orchestration/agents/fault_tolerance.py) | Python | -307 | -58 | -91 | -456 |
| [orchestration/agents/load\_balancer.py](/orchestration/agents/load_balancer.py) | Python | -219 | -46 | -77 | -342 |
| [orchestration/agents/multimodal\_config.py](/orchestration/agents/multimodal_config.py) | Python | -106 | -3 | -18 | -127 |
| [orchestration/agents/registry.py](/orchestration/agents/registry.py) | Python | -212 | -19 | -50 | -281 |
| [orchestration/manager/\_\_init\_\_.py](/orchestration/manager/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [orchestration/manager/orchestration\_manager.py](/orchestration/manager/orchestration_manager.py) | Python | -404 | -79 | -121 | -604 |
| [orchestration/message\_bus/\_\_init\_\_.py](/orchestration/message_bus/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [orchestration/message\_bus/message\_formats.py](/orchestration/message_bus/message_formats.py) | Python | -197 | -5 | -36 | -238 |
| [orchestration/message\_bus/message\_handler.py](/orchestration/message_bus/message_handler.py) | Python | -204 | -21 | -52 | -277 |
| [orchestration/message\_bus/rabbitmq\_wrapper.py](/orchestration/message_bus/rabbitmq_wrapper.py) | Python | -267 | -29 | -58 | -354 |
| [orchestration/monitoring/\_\_init\_\_.py](/orchestration/monitoring/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [orchestration/monitoring/logger.py](/orchestration/monitoring/logger.py) | Python | -106 | -15 | -34 | -155 |
| [orchestration/monitoring/metrics.py](/orchestration/monitoring/metrics.py) | Python | -204 | -14 | -51 | -269 |
| [orchestration/monitoring/tracing.py](/orchestration/monitoring/tracing.py) | Python | -153 | -4 | -32 | -189 |
| [orchestration/tests/\_\_init\_\_.py](/orchestration/tests/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [orchestration/workflow/\_\_init\_\_.py](/orchestration/workflow/__init__.py) | Python | 0 | 0 | -1 | -1 |
| [orchestration/workflow/end\_to\_end\_workflows.py](/orchestration/workflow/end_to_end_workflows.py) | Python | -259 | -37 | -73 | -369 |
| [orchestration/workflow/enhanced\_workflows.py](/orchestration/workflow/enhanced_workflows.py) | Python | -957 | -111 | -232 | -1,300 |
| [orchestration/workflow/error\_handler.py](/orchestration/workflow/error_handler.py) | Python | -233 | -21 | -54 | -308 |
| [orchestration/workflow/error\_recovery.py](/orchestration/workflow/error_recovery.py) | Python | -1,126 | -172 | -274 | -1,572 |
| [orchestration/workflow/integrated\_response\_workflow.py](/orchestration/workflow/integrated_response_workflow.py) | Python | -344 | -35 | -87 | -466 |
| [orchestration/workflow/multimodal\_workflows.py](/orchestration/workflow/multimodal_workflows.py) | Python | -383 | -22 | -60 | -465 |
| [orchestration/workflow/standard\_workflows.py](/orchestration/workflow/standard_workflows.py) | Python | -265 | -7 | -26 | -298 |
| [orchestration/workflow/visualization\_handler.py](/orchestration/workflow/visualization_handler.py) | Python | -352 | -64 | -94 | -510 |
| [orchestration/workflow/visualization\_workflows.py](/orchestration/workflow/visualization_workflows.py) | Python | -143 | -11 | -14 | -168 |
| [orchestration/workflow/workflow\_definition.py](/orchestration/workflow/workflow_definition.py) | Python | -185 | -7 | -37 | -229 |
| [orchestration/workflow/workflow\_engine.py](/orchestration/workflow/workflow_engine.py) | Python | -958 | -142 | -272 | -1,372 |
| [orchestration/workflow/workflow\_registry.py](/orchestration/workflow/workflow_registry.py) | Python | -353 | -28 | -64 | -445 |
| [scripts/\_\_init\_\_.py](/scripts/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [scripts/db\_backup.py](/scripts/db_backup.py) | Python | 153 | 21 | 46 | 220 |
| [scripts/db\_restore.py](/scripts/db_restore.py) | Python | 129 | 17 | 32 | 178 |
| [scripts/initialize\_message\_bus.py](/scripts/initialize_message_bus.py) | Python | 175 | 17 | 39 | 231 |
| [scripts/make\_advanced\_ocr\_executable.sh](/scripts/make_advanced_ocr_executable.sh) | Shell Script | 2 | 2 | 1 | 5 |
| [scripts/make\_executable.sh](/scripts/make_executable.sh) | Shell Script | 1 | 1 | 1 | 3 |
| [scripts/make\_ocr\_executable.sh](/scripts/make_ocr_executable.sh) | Shell Script | 2 | 2 | 1 | 5 |
| [scripts/setup\_database.py](/scripts/setup_database.py) | Python | 77 | 12 | 23 | 112 |
| [scripts/test\_advanced\_image\_processing.py](/scripts/test_advanced_image_processing.py) | Python | 227 | 45 | 73 | 345 |
| [scripts/test\_agent\_communication.py](/scripts/test_agent_communication.py) | Python | 208 | 36 | 60 | 304 |
| [scripts/test\_agent\_registry.py](/scripts/test_agent_registry.py) | Python | 105 | 28 | 48 | 181 |
| [scripts/test\_core\_components.py](/scripts/test_core_components.py) | Python | 68 | 12 | 24 | 104 |
| [scripts/test\_database.py](/scripts/test_database.py) | Python | 145 | 28 | 47 | 220 |
| [scripts/test\_handwriting\_recognition.py](/scripts/test_handwriting_recognition.py) | Python | 90 | 13 | 26 | 129 |
| [scripts/test\_integration.py](/scripts/test_integration.py) | Python | 541 | 58 | 138 | 737 |
| [scripts/test\_logging.py](/scripts/test_logging.py) | Python | 69 | 12 | 32 | 113 |
| [scripts/test\_math\_knowledge\_agent.py](/scripts/test_math_knowledge_agent.py) | Python | 135 | 8 | 33 | 176 |
| [scripts/test\_math\_ocr\_enhancement.py](/scripts/test_math_ocr_enhancement.py) | Python | 108 | 14 | 32 | 154 |
| [scripts/test\_message\_bus.py](/scripts/test_message_bus.py) | Python | 140 | 20 | 40 | 200 |
| [scripts/test\_mistral\_integration.py](/scripts/test_mistral_integration.py) | Python | 138 | 14 | 39 | 191 |
| [scripts/test\_orchestration.py](/scripts/test_orchestration.py) | Python | 300 | 27 | 79 | 406 |
| [scripts/test\_visualization.py](/scripts/test_visualization.py) | Python | 139 | 5 | 24 | 168 |
| [scripts/validate\_sprint1.py](/scripts/validate_sprint1.py) | Python | 123 | 20 | 38 | 181 |

[Summary](results.md) / [Details](details.md) / [Diff Summary](diff.md) / Diff Details
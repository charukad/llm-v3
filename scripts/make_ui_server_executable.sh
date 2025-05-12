#!/bin/bash
chmod +x scripts/start_test_ui_server.js
echo '#!/usr/bin/env node' | cat - scripts/start_test_ui_server.js > temp && mv temp scripts/start_test_ui_server.js
echo "Script made executable. Run with: ./scripts/start_test_ui_server.js"

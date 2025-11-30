빠른 시작
# 1. 세션 초기화
/mas-init

# 2. 전체 자동 실행 (Phase 1-5)
/mas-run

# 3. 품질 검증
/mas-verify all

# 4. 커밋
/mas-git commit

단계별 실행
# Phase별 실행
/mas-phase 1    # Setup & Config
/mas-phase 2    # Base Agent
/mas-phase 3    # Coding Agent
/mas-phase 4    # Orchestrator
/mas-phase 5    # Integration

TDD 사이클
# 모듈별 TDD
/mas-tdd config full
/mas-tdd base_agent full
/mas-tdd coding_agent full
/mas-tdd orchestrator full
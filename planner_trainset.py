import dspy

from workflow import RetrievalTask, RetrievalTaskList

TODAY = "2026-02-10"

planner_trainset = [
    # =========================================================================
    # Group 1: Temporal & Evolution (時間維度：歷史與演變)
    # 目的：訓練模型識別 "history", "introduced", "change" 並切換至 Global Mode。
    # =========================================================================
    dspy.Example(
        query="In which year was the CCFtap scenario first introduced?",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="global",  # 關鍵：詢問 "引入時間" 屬於演變史
                    target_date=None,
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",  # 根據 "CCFtap" 判斷為 C2C
                    rewritten_query="CCFtap scenario introduction",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="How has the AEB VRU scoring changed over the years?",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="global",  # 關鍵：詢問 "變化 (changed)" 且無特定版本
                    target_date=None,
                    target_version=None,
                    protocol_type="Assessment Protocol",  # 關鍵：詢問 "scoring" -> Assessment
                    system_domain="Vulnerable Road User",  # 關鍵：詢問 "VRU"
                    rewritten_query="AEB VRU scoring criteria history",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    # =========================================================================
    # Group 2: Precision & Explicit Version (精確模式：指定版本)
    # 目的：訓練模型提取 "vX.X" 版本號並鎖定 Precision Mode。
    # =========================================================================
    dspy.Example(
        query="What is the test speed for CCRs in v4.3.1?",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="4.3.1",  # 明確提取版本
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",  # CCRs -> C2C
                    rewritten_query="CCRs test speed",  # 簡單名詞片語
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="Requirements for Child Presence Detection in version 11.0.",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="11.0",
                    protocol_type="Test Protocol",
                    system_domain=None,  # CPD 不屬於 C2C 或 VRU，保持 None
                    rewritten_query="Child Presence Detection requirements",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    # =========================================================================
    # Group 3: Implicit Today (精確模式：當下/預設)
    # 目的：訓練模型處理 "current", "latest" 或未指定時間的情況，使用 input 的 today。
    # =========================================================================
    dspy.Example(
        query="What are the current requirements for rescue sheets?",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=TODAY,  # "current" -> 使用當前日期
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain=None,  # Rescue sheet 不屬於 C2C/VRU
                    rewritten_query="rescue sheet requirements",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    # =========================================================================
    # Group 4: Domain Mapping Logic (領域映射訓練)
    # 目的：嚴格測試 CC* -> C2C, CP*/CB* -> VRU 的映射規則。
    # =========================================================================
    dspy.Example(
        query="Describe the target path for CPTA scenarios.",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",  # 未指定時間，預設查當下
                    target_date=TODAY,
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",  # CPTA -> VRU
                    rewritten_query="CPTA scenario target path",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="Tolerances for CCRb vehicle speed.",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=TODAY,
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",  # CCRb -> C2C
                    rewritten_query="CCRb vehicle speed tolerances",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    # =========================================================================
    # Group 5: Protocol Discrimination (協定類型辨識)
    # 目的：區分 "Test" (物理/速度) 與 "Assessment" (分數/點數)。
    # =========================================================================
    dspy.Example(
        query="How many points are awarded for ELK?",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=TODAY,
                    target_version=None,
                    protocol_type="Assessment Protocol",  # "points" -> Assessment
                    system_domain=None,  # ELK 不屬於 C2C/VRU
                    rewritten_query="ELK point awarding criteria",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="Star rating calculation method for occupant status.",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=TODAY,
                    target_version=None,
                    protocol_type="Assessment Protocol",  # "Star rating" -> Assessment
                    system_domain=None,
                    rewritten_query="occupant status star rating calculation",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    # =========================================================================
    # Group 6: Comparative Tasks (版本比較 - 複雜拆解)
    # 目的：訓練模型將 "Comparison" 拆解為兩個獨立的 Task，並去除比較詞。
    # =========================================================================
    dspy.Example(
        query="Compare the test speed of CCRm between v3.0 and v4.1.",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                # Task 1: v3.0
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="3.0",
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="CCRm test speed",  # 移除 "Compare", "between"
                ),
                # Task 2: v4.1
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="4.1",
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="CCRm test speed",
                ),
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the difference in pedestrian impact point between 2020 and 2023?",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                # Task 1: 2020
                RetrievalTask(
                    mode="precision",
                    target_date="2020-01-01",  # 提取年份轉為日期
                    target_version=None,
                    protocol_type="Test Protocol",  # 未提分數，預設 Test (Impact points指撞擊點)
                    system_domain="Vulnerable Road User",  # Pedestrian -> VRU
                    rewritten_query="pedestrian impact point definition",  # 移除 "difference"
                ),
                # Task 2: 2023
                RetrievalTask(
                    mode="precision",
                    target_date="2023-01-01",
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",
                    rewritten_query="pedestrian impact point definition",
                ),
            ]
        ),
    ).with_inputs("query", "today"),
    # =========================================================================
    # Group 7: Negative Constraints (負向約束測試)
    # 目的：測試模型是否能忽略 "New", "Old", "Difference" 等干擾詞。
    # =========================================================================
    dspy.Example(
        query="What are the new clarifications for CBna in v9.0?",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="9.0",
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",  # CBna -> VRU
                    rewritten_query="CBna clarifications",  # 移除 "new"
                )
            ]
        ),
    ).with_inputs("query", "today"),
    # =========================================================================
    # Group 8: Multi-Domain Query (跨領域查詢)
    # 目的：測試單一 Query 拆解為不同 Domain 的 Task。
    # =========================================================================
    dspy.Example(
        query="Test scenarios for both Car-to-Car and Pedestrian systems.",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=TODAY,
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="Car-to-Car test scenarios",
                ),
                RetrievalTask(
                    mode="precision",
                    target_date=TODAY,
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",
                    rewritten_query="Pedestrian test scenarios",
                ),
            ]
        ),
    ).with_inputs("query", "today"),
    # =========================================================================
    # Group 9: Edge Case - Ambiguous Domain (邊界案例)
    # 目的：測試非標準關鍵字下的 Domain 判斷 (應為 None)
    # =========================================================================
    dspy.Example(
        query="HMI requirements for Speed Assist Systems.",
        today=TODAY,
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=TODAY,
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain=None,  # SAS 不屬於 C2C/VRU
                    rewritten_query="Speed Assist System HMI requirements",
                )
            ]
        ),
    ).with_inputs("query", "today"),
]

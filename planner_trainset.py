import dspy

from workflow import RetrievalTask, RetrievalTaskList

planner_trainset = [
    dspy.Example(
        query="What are the initial test speeds for the CCRs scenario in the 2023 Test Protocol?",
        today="2025-05-20",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date="2023-01-01",  # 從 "2023" 提取
                    target_version=None,  # 未指定具體版本號
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",  # CCRs 屬於 C2C
                    rewritten_query="CCRs initial test speed specification",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What are the scoring requirements for the CPNCO-50 scenario in VRU Assessment Protocol v4.5?",
        today="2026-02-11",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,  # 已指定具體版本，日期設為 None
                    target_version="4.5",  # 明確提取版本號
                    protocol_type="Assessment Protocol",
                    system_domain="Vulnerable Road User",  # CPNCO 屬於 VRU
                    rewritten_query="CPNCO-50 scoring requirements points calculation",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="Compare the target vehicle speed requirements for the CCRb scenario between Test Protocol v2.0.1 and v4.3.",
        today="2026-03-15",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="2.0.1",
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="CCRb target vehicle speed requirements",
                ),
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="4.3",
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="CCRb target vehicle speed requirements",
                ),
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="When was the oncoming vehicle scenario added to the test scenarios?",
        today="2026-05-20",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="global",  # 詢問 "When ... added" -> Global
                    target_date=None,  # Global 模式下不設特定日期
                    target_version=None,  # Global 模式下不設特定版本
                    protocol_type="Test Protocol",  # 詢問 "test scenarios"
                    system_domain="Car-to-Car",  # 對向車屬於 C2C
                    rewritten_query="Oncoming vehicle test scenario introduction timeline",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the allowable lateral path error for the VUT in AEB VRU test scenarios active today?",
        today="2026-02-11",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date="2026-02-11",  # 映射為 "Today"
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",
                    rewritten_query="AEB VRU VUT lateral path error tolerance",  # 聚焦於 Lateral Path
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What are the VUT and GVT test speeds for the CCFtap scenario in the 2023 C2C Test Protocol?",
        today="2026-06-12",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date="2023-01-01",  # 從 "2023" 提取年份
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",  # CCFtap 屬於 C2C
                    rewritten_query="CCFtap VUT GVT test speed specification",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the ambient lighting specification for night tests in VRU Test Protocol v4.4?",
        today="2026-07-20",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,  # 已指定版本，日期設為 None
                    target_version="4.4",  # 提取版本號
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",  # 夜間測試屬於 VRU
                    rewritten_query="Night test ambient lighting specification",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the Radar Cross Section (RCS) requirement for the Global Vehicle Target (GVT) in the current C2C Test Protocol?",
        today="2020-08-15",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date="2020-08-15",  # "current" -> Today
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",  # GVT 屬於 C2C
                    rewritten_query="Global Vehicle Target RCS radar cross section specification",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the exact definition and calculation method of a 50% impact overlap for the CCRm scenario in C2C Test Protocol v4.3?",
        today="2026-09-01",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="4.3",
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="CCRm 50% impact overlap definition calculation method",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the pedestrian dummy lateral starting distance for the CPFA scenario at a 20 km/h VUT test speed in VRU Test Protocol v4.5.1?",
        today="2026-10-05",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="4.5.1",
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",
                    rewritten_query="CPFA 20 km/h VUT speed pedestrian dummy lateral starting distance",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the specified crossing speed of the Euro NCAP Bicyclist Target (EBT) in the CBNA scenario according to VRU Test Protocol v4.5?",
        today="2026-11-10",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="4.5",
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",
                    rewritten_query="CBNA Target crossing speed specification",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the required VUT test mass condition for performing AEB tests in the 2023 C2C Test Protocol?",
        today="2026-12-01",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date="2023-01-01",
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="AEB VUT test mass condition requirement",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the required Forward Collision Warning (FCW) Time-to-Collision (TTC) to receive full points in the CCRm scenario in the 2023 Assessment Protocol?",
        today="2027-01-15",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date="2023-01-01",
                    target_version=None,
                    protocol_type="Assessment Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="CCRm Forward Collision Warning FCW Time-to-Collision TTC full points requirement",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the VUT reverse test speed range for the CPRA scenario in VRU Test Protocol v4.4?",
        today="2027-02-15",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date=None,
                    target_version="4.4",
                    protocol_type="Test Protocol",
                    system_domain="Vulnerable Road User",
                    rewritten_query="CPRA VUT reverse test speed range specification",
                )
            ]
        ),
    ).with_inputs("query", "today"),
    dspy.Example(
        query="What is the required Peak Braking Coefficient (PBC) for the test track surface in the 2023 C2C Test Protocol?",
        today="2027-04-10",
        plan=RetrievalTaskList(
            tasks=[
                RetrievalTask(
                    mode="precision",
                    target_date="2023-01-01",
                    target_version=None,
                    protocol_type="Test Protocol",
                    system_domain="Car-to-Car",
                    rewritten_query="test track surface Peak Braking Coefficient PBC requirement",
                )
            ]
        ),
    ).with_inputs("query", "today"),
]

print(len(planner_trainset))

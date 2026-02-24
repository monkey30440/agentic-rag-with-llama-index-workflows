import dspy

from planner_trainset import planner_trainset

synthesizer_trainset = [
    dspy.Example(
        # Query: "What are the initial test speeds for the CCRs scenario in the 2023 Test Protocol?"
        query=planner_trainset[0].query,
        context=[
            "[Doc 1] Euro NCAP AEB C2C Test Protocol v4.1 (Active in 2023). Section 7.1: Car-to-Car Rear Stationary (CCRs). For AEB system evaluations, the VUT (Vehicle Under Test) initial test speeds are defined from 10 km/h up to 50 km/h.",
            "[Doc 2] Section 7.2: Car-to-Car Rear Moving (CCRm). The VUT initial test speed shall be tested from 30 km/h to 80 km/h with the GVT traveling at a constant speed of 20 km/h.",
            "[Doc 3] Section 7.1.2: CCRs for FCW. Forward Collision Warning system tests in the CCRs scenario are conducted at VUT speeds starting from 30 km/h up to 80 km/h.",
        ],
        answer="According to the 2023 C2C Test Protocol, the initial test speeds for the CCRs scenario are:\n- AEB systems: 10 km/h to 50 km/h [Doc 1].\n- FCW systems: 30 km/h to 80 km/h [Doc 3].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What are the scoring requirements for the CPNCO-50 scenario in VRU Assessment Protocol v4.5?"
        query=planner_trainset[1].query,
        context=[
            "[Doc 1] Euro NCAP Assessment Protocol - Vulnerable Road User Protection v4.5. Section 5.2.2: CPNCO-50 (Car-to-Pedestrian Nearside Child Obstructed 50%). For the CPNCO-50 scenario, the maximum points are awarded if the test vehicle achieves a relative impact speed of 0 km/h (avoidance). If the impact speed is greater than 20 km/h, 0 points are awarded. A sliding scale applies for impact speeds between 0 km/h and 20 km/h.",
            "[Doc 2] Euro NCAP Assessment Protocol - Vulnerable Road User Protection v4.5. Section 5.2.3: CPNCO-25. In the 25% overlap scenario, point deduction occurs if the AEB activation time is less than 0.8 seconds before impact.",
            "[Doc 3] Euro NCAP Test Protocol - AEB VRU systems v4.5. The child dummy used for CPNCO tests represents a 6-year-old and shall be positioned behind the obstruction vehicles.",
        ],
        answer="According to the VRU Assessment Protocol v4.5, the scoring requirements for the CPNCO-50 scenario are:\n- Maximum points: Impact speed of 0 km/h (complete avoidance) [Doc 1].\n- Zero points: Impact speed greater than 20 km/h [Doc 1].\n- Sliding scale: Impact speeds between 0 km/h and 20 km/h [Doc 1].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "Compare the target vehicle speed requirements for the CCRb scenario between Test Protocol v2.0.1 and v4.3."
        query=planner_trainset[2].query,
        context=[
            "[Doc 1] Euro NCAP AEB C2C Test Protocol v2.0.1. Section 6.2: CCRb (Car-to-Car Rear Braking). The Global Vehicle Target (GVT) shall be driven at a constant speed of 50 km/h in front of the VUT. The VUT shall also maintain an initial speed of 50 km/h with a 12m headway or 40m headway.",
            "[Doc 2] Euro NCAP C2C Test Protocol v4.3. Section 7.3: CCRb. The target vehicle (GVT) and the VUT shall be driven at a constant speed of 50 km/h. Tests are performed at both 12m and 40m headways. The GVT deceleration is set to -2 m/s² and -6 m/s².",
            "[Doc 3] Euro NCAP Test Protocol v3.0. For the CCRm scenario, the GVT maintains 20 km/h.",
        ],
        answer="Based on the comparison between Test Protocol v2.0.1 and v4.3 for the CCRb scenario, the target vehicle speed requirements remain identical. In both v2.0.1 [Doc 1] and v4.3 [Doc 2], the target vehicle (GVT) is required to be driven at a constant speed of 50 km/h.",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "When was the oncoming vehicle scenario added to the test scenarios?"
        query=planner_trainset[3].query,
        context=[
            "[Doc 1] Euro NCAP 2025 Roadmap. The assessment of AEB systems was first introduced in 2014 for rear-end crashes. For the 2020 update, the AEB Car-to-Car test scenarios are expanded to include junction assist (Turn Across Path). By 2023, direct oncoming Head-On scenarios will be added.",
            "[Doc 2] Euro NCAP C2C Test Protocol v3.0. Active from January 2020. Section 8: Car-to-Car Front turn-across-path (CCFtap). This new scenario tests the VUT turning across the path of an oncoming GVT.",
            "[Doc 3] Euro NCAP C2C Test Protocol v4.0. Active from January 2023. Section 9: Head-On scenarios. Introduces CCFhol (Car-to-Car Front Head-On drift-over-lane) and CCFhos (straight) to test oncoming vehicle collisions.",
        ],
        answer="The oncoming vehicle scenarios were introduced to Euro NCAP testing in two phases:\n- The Turn-Across-Path (CCFtap) scenario, where the test vehicle turns across an oncoming vehicle's path, was added in 2020 [Doc 1, Doc 2].\n- The direct Head-On scenarios (CCFhol and CCFhos) were introduced in 2023 [Doc 1, Doc 3].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the allowable lateral path error for the VUT in AEB VRU test scenarios active today?"
        query=planner_trainset[4].query,
        context=[
            "[Doc 1] Euro NCAP Test Protocol - AEB/LSS VRU systems v4.4 (Active 2026). Section 4.2.3: The required lateral path error for the VUT (Vehicle Under Test) is ±0.05m during the approach to the collision point.",
            "[Doc 2] Section 4.2.4: The yaw rate error shall not exceed ±1.0 deg/s. The steering wheel angle velocity shall be maintained below 15 deg/s.",
            "[Doc 3] Euro NCAP Test Protocol - C2C systems v4.3. The lateral path error for VUT in Car-to-Car scenarios is ±0.1m.",
        ],
        answer="According to the AEB VRU Test Protocol active today (v4.4), the allowable lateral path error for the VUT (Vehicle Under Test) is ±0.05m [Doc 1].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What are the VUT and GVT test speeds for the CCFtap scenario in the 2023 C2C Test Protocol?"
        query=planner_trainset[5].query,
        context=[
            "[Doc 1] Euro NCAP AEB C2C Test Protocol v4.1 (Active in 2023). Section 8.1: Car-to-Car Front turn-across-path (CCFtap). The VUT test speeds shall be 10, 15, and 20 km/h.",
            "[Doc 2] Section 8.2: For the CCFtap scenario, the GVT (Global Vehicle Target) test speeds are defined as 30, 45, and 55 km/h. The impact point is evaluated at 50% overlap.",
            "[Doc 3] Euro NCAP C2C Test Protocol v3.0. In older versions, CCFtap only tested GVT speeds up to 50 km/h.",
        ],
        answer="According to the 2023 C2C Test Protocol, the test speeds for the CCFtap scenario are:\n- VUT (Vehicle Under Test): 10, 15, and 20 km/h [Doc 1].\n- GVT (Global Vehicle Target): 30, 45, and 55 km/h [Doc 2].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the ambient lighting specification for night tests in VRU Test Protocol v4.4?"
        query=planner_trainset[6].query,
        context=[
            "[Doc 1] Euro NCAP Test Protocol - AEB/LSS VRU systems v4.4. Section 3.1.2: Night tests. Testing shall be performed in complete darkness. The ambient illumination must not exceed 1.0 lux.",
            "[Doc 2] Section 3.1.3: Street light condition. For tests requiring street lamps, the uniform illuminance shall be 19 lux.",
            "[Doc 3] Hardware Specifications v3.0. The lux meter used for verification must be calibrated annually.",
        ],
        answer="According to the VRU Test Protocol v4.4, the ambient illumination for night tests in complete darkness must not exceed 1.0 lux [Doc 1].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the Radar Cross Section (RCS) requirement for the Global Vehicle Target (GVT) in the current C2C Test Protocol?"
        query=planner_trainset[7].query,
        context=[
            "[Doc 1] Hardware Specifications for Euro NCAP Global Vehicle Target (GVT). Section 4.2: Radar Cross Section (RCS). The GVT must exhibit a radar signature equivalent to a typical C-segment passenger car.",
            "[Doc 2] Section 4.3: The required RCS for the GVT shall be within the range of 10 dBsm to 15 dBsm when measured at a 77 GHz frequency from a distance of 30 meters.",
            "[Doc 3] Euro NCAP Test Protocol - C2C systems v4.3. The GVT must be properly inflated and visually intact before the RCS verification test.",
        ],
        answer="According to the current test specifications, the required Radar Cross Section (RCS) for the Global Vehicle Target (GVT) is between 10 dBsm and 15 dBsm at a 77 GHz frequency [Doc 2].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the exact definition and calculation method of a 50% impact overlap for the CCRm scenario in C2C Test Protocol v4.3?"
        query=planner_trainset[8].query,
        context=[
            "[Doc 1] Euro NCAP C2C Test Protocol v4.3. Section 5.3: Impact Overlap Definition. For the CCRm scenario, a 50% impact overlap is defined as aligning the longitudinal centerline of the VUT with the outermost lateral edge of the GVT.",
            "[Doc 2] Section 5.3.1: Calculation Method. The overlap percentage is calculated as: Overlap (%) = ((VUT width / 2) / VUT width) * 100. The VUT width excludes side mirrors.",
            "[Doc 3] Section 5.4: 100% impact overlap is defined as aligning the centerline of the VUT perfectly with the centerline of the GVT.",
        ],
        answer="According to the C2C Test Protocol v4.3, the specifications for a 50% impact overlap in the CCRm scenario are:\n- Definition: Aligning the longitudinal centerline of the VUT with the outermost lateral edge of the GVT [Doc 1].\n- Calculation Method: Overlap (%) = ((VUT width / 2) / VUT width) * 100 (excluding side mirrors) [Doc 2].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the pedestrian dummy lateral starting distance for the CPFA scenario at a 20 km/h VUT test speed in VRU Test Protocol v4.5.1?"
        query=planner_trainset[9].query,
        context=[
            "[Doc 1] Euro NCAP Test Protocol - AEB/LSS VRU systems v4.5.1. Section 7.1: Car-to-Pedestrian Farside Adult (CPFA). The dummy lateral starting distance depends on the VUT test speed.",
            "[Doc 2] Section 7.1.2: If the VUT test speed is 20 km/h, the pedestrian dummy lateral starting distance shall be 4.0 meters from the VUT longitudinal centerline.",
            "[Doc 3] Section 7.1.3: For a VUT test speed of 30 km/h or higher, the lateral starting distance increases to 6.0 meters to allow sufficient dummy acceleration time.",
        ],
        answer="According to the VRU Test Protocol v4.5.1, for the CPFA scenario at a 20 km/h VUT test speed, the pedestrian dummy lateral starting distance is 4.0 meters from the VUT longitudinal centerline [Doc 2].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the specified crossing speed of the Euro NCAP Bicyclist Target (EBT) in the CBNA scenario according to VRU Test Protocol v4.5?"
        query=planner_trainset[10].query,
        context=[
            "[Doc 1] Euro NCAP Test Protocol - AEB/LSS VRU systems v4.5. Section 7.3: Car-to-Bicyclist Nearside Adult (CBNA). The test requires the VUT to approach the Euro NCAP Bicyclist Target (EBT) which is crossing the VUT path from the nearside.",
            "[Doc 2] Section 7.3.2: The crossing speed of the EBT shall be 15 km/h ± 0.5 km/h. The impact point is set at 50% of the VUT width.",
            "[Doc 3] Section 7.4: Car-to-Bicyclist Farside Adult (CBFA). The EBT crossing speed for the farside scenario shall be 20 km/h.",
        ],
        answer="According to the VRU Test Protocol v4.5, the specified crossing speed of the Euro NCAP Bicyclist Target (EBT) in the CBNA scenario is 15 km/h ± 0.5 km/h [Doc 2].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the required VUT test mass condition for performing AEB tests in the 2023 C2C Test Protocol?"
        query=planner_trainset[11].query,
        context=[
            "[Doc 1] Euro NCAP C2C Test Protocol v4.1 (2023). Section 2.1: VUT preparation. The vehicle under test shall be prepared for evaluation.",
            "[Doc 2] Section 2.1.3: Test Mass. The required VUT test mass condition for performing AEB tests is the Unladen Mass plus 200 kg. This additional 200 kg represents the weight of the driver and the data acquisition instrumentation.",
            "[Doc 3] Section 2.1.4: The fuel tank shall be filled to at least 90% of its total capacity.",
        ],
        answer="According to the 2023 C2C Test Protocol, the required VUT test mass condition for performing AEB tests is the Unladen Mass plus 200 kg [Doc 2].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the required Forward Collision Warning (FCW) Time-to-Collision (TTC) to receive full points in the CCRm scenario in the 2023 Assessment Protocol?"
        query=planner_trainset[12].query,
        context=[
            "[Doc 1] Euro NCAP Assessment Protocol - Safety Assist v10.0 (2023). Section 4.2: CCRm scenario scoring.",
            "[Doc 2] Section 4.2.3: For Forward Collision Warning (FCW) evaluation in the CCRm scenario, maximum points (full points) are awarded if the system issues an alert at a Time-to-Collision (TTC) of ≥ 2.1 seconds.",
            "[Doc 3] Section 4.2.4: If the FCW alert is issued between 1.5 seconds and 2.1 seconds, a sliding scale point deduction is applied.",
        ],
        answer="According to the 2023 Assessment Protocol, the required Forward Collision Warning (FCW) Time-to-Collision (TTC) to receive full points in the CCRm scenario is ≥ 2.1 seconds [Doc 2].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the VUT reverse test speed range for the CPRA scenario in VRU Test Protocol v4.4?"
        query=planner_trainset[13].query,
        context=[
            "[Doc 1] Euro NCAP Test Protocol - AEB/LSS VRU systems v4.4. Section 8.1: Car-to-Pedestrian Reverse Adult (CPRA).",
            "[Doc 2] Section 8.1.2: VUT Test Speeds. For the CPRA scenario, the VUT reverse test speed range shall be from 4 km/h to 14 km/h. Tests are conducted in increments of 2 km/h.",
            "[Doc 3] Section 8.2: Car-to-Pedestrian Reverse Child (CPRC). The test speed range is identical to the CPRA scenario.",
        ],
        answer="According to the VRU Test Protocol v4.4, the VUT reverse test speed range for the CPRA scenario is from 4 km/h to 14 km/h [Doc 2].",
    ).with_inputs("query", "context"),
    dspy.Example(
        # Query: "What is the required Peak Braking Coefficient (PBC) for the test track surface in the 2023 C2C Test Protocol?"
        query=planner_trainset[14].query,
        context=[
            "[Doc 1] Euro NCAP C2C Test Protocol v4.1 (2023). Section 3.1: Test Track Conditions.",
            "[Doc 2] Section 3.1.2: Peak Braking Coefficient (PBC). The required PBC for the test track surface shall be 0.9. The measurement shall be performed using an ASTM E1136 standard reference test tire.",
            "[Doc 3] Section 3.1.3: The track gradient must not exceed 1% in the longitudinal direction.",
        ],
        answer="According to the 2023 C2C Test Protocol, the required Peak Braking Coefficient (PBC) for the test track surface is 0.9 [Doc 2].",
    ).with_inputs("query", "context"),
]

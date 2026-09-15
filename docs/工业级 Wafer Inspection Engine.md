                         Wafer
                           │
                           ↓
                  Wafer Classification
                           │
          ┌────────────────┼─────────────────┐
          │                │                 │
          ↓                ↓                 ↓
      No Pattern       Patterned        Macro Inspection
          │                │                 │
          ↓                ↓                 ↓
   Background Model   Pattern Analysis   Low Frequency
          │                │             Background
          ↓                │                 │
   Bright/Dark          ┌──┴────┐            ↓
   Detection            │       │       Macro Defect
                        ↓       ↓
                     D2D     Single-Die
                        │       │
                        └──┬────┘
                           ↓
                    Reference Builder
                           │
                    ┌──────┴──────┐
                    ↓             ↓
                 Golden       Multi-Die
                 Reference     Consensus
                    │             │
                    └──────┬──────┘
                           ↓
                    Registration
                           ↓
                   Normalization
                           ↓
                  Residual Generation
                           ↓
                 Candidate Detection
                           ↓
                  Defect Classification
                           ↓
                     Final Defects
					 
					 



					 
                   InspectionEngine
                         │
              ┌──────────┴──────────┐
              │                     │
       Wafer-Level             Die-Level
       Inspection              Inspection
              │                     │
       Macro Defect           ┌─────┴─────┐
                              │           │
                             D2D       Single-Die
                              │           │
                              └─────┬─────┘
                                    ↓
                             Reference Engine
                                    │
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
                 Golden        Multi-Die        Internal
                 Reference      Consensus       Pattern
                    │               │               │
                    └───────────────┼───────────────┘
                                    ↓
                              Registration
                                    ↓
                              Normalization
                                    ↓
                              Residual Map
                                    ↓
                           Candidate Generation
                                    ↓
                           Defect Classification
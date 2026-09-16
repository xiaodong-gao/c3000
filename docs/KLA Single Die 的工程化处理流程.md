                Input Single Die
                       │
                       ▼
              ROI / Die Boundary
                       │
                       ▼
             Image Preprocessing
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
   Intensity       Noise         Background
   Normalize       Estimate      Correction
          │            │            │
          └────────────┼────────────┘
                       ▼
                  Alignment
                       │
                       ▼
              Multi-Scale Analysis
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Cell 1        Cell 2       Cell 3
       Small         Medium        Large
          │            │            │
          ▼            ▼            ▼
      Feature       Feature       Feature
       Image          Image         Image
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Feature Fusion
                       │
                       ▼
              Candidate Detection
                       │
                       ▼
               Morphology / Blob
                       │
                       ▼
               Candidate Filtering
                       │
                       ▼
              Defect Classification
                       │
                       ▼
              Final Defect Output
			  
			  
				Input
				 │
				 ▼
				IntensityNormalize
				 │
				 ▼
				NoiseEstimation
				 │
				 ▼
				Multi-scale Feature
				 │
				 ├──────────────┐
				 ▼              ▼
				Histogram      Local Statistics
				 │              │
				 ▼              ▼
				Threshold      Difference
				 │              │
				 └──────┬───────┘
						▼
				   Candidate Map
						│
						▼
				   Morphology
						│
						▼
				   Blob Analysis
						│
						▼
				   Defect Decision
				   
	


	
				   Inspection Engine
                        │
              ┌─────────┴─────────┐
              │                   │
          Reference             Feature
          Generator             Generator
              │                   │
       ┌──────┼───────┐           │
       │      │       │           │
      D2D   Golden   DB          Multi-scale
       │      │       │           │
       └──────┼───────┘           │
              │                   │
              └─────────┬─────────┘
                        ▼
                 Alignment Engine
                        │
                        ▼
                 Difference Engine
                        │
                        ▼
                Candidate Generator
                        │
                        ▼
                  Blob Analysis
                        │
                        ▼
                 Defect Classifier
				 
				 
				 
				 
				  ┌──────────────────┐
                  │     Single Die   │
                  │      Image       │
                  └────────┬─────────┘
                           ↓
                 ┌──────────────────┐
                 │ IntensityNormalize│
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Noise Estimation │
                 └────────┬─────────┘
                          ↓
                ┌────────────────────┐
                │ Background Removal │
                └──────────┬─────────┘
                           ↓
                 ┌──────────────────┐
                 │ Alignment Engine │
                 │ Phase/ECC        │
                 └────────┬─────────┘
                          ↓
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
   Small Cell          Medium Cell       Large Cell
       ↓                  ↓                  ↓
 Local Feature        Local Feature      Local Feature
       ↓                  ↓                  ↓
   Threshold          Threshold          Threshold
       └──────────────────┼──────────────────┘
                          ↓
                  Feature Fusion
                          ↓
                  Candidate Map
                          ↓
                 Morphology Filter
                          ↓
                   Blob Analysis
                          ↓
               Size / Shape / Intensity
                          ↓
                   Defect Candidate
                          ↓
                 Defect Classification
                          ↓
                ┌─────────────────┐
                │ Final Defect Map│
                └─────────────────┘
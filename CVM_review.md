1. Paper summary
   (Briefly describe the paper and its contribution.  Please give your assessment of the novelty and magnitude of the paper's contribution.)
    This paper introduce a monocular 360 degree multi-task learning architechture for estimating both depth and surface normals. The fusion module and the multi-scale spherical decoder efficiently enhance the depth estimation and improve the accuracy of surface normal estimation. 

2. Organization and clarity of presentation

2.1) Is the exposition clear? Is the title suitable? Is the paper too short or too long? How could it be improved?
    The paper is well-organized and clear. The title is suitable and concise. The paper is not too short or too long. The exposition is clear and easy to follow. 

2.2) Is the paper described in sufficient detail for a competent researcher to be able to reproduce the results?
    Yes, the paper is well-described and provides sufficient detail for a competent researcher to reproduce the results.

2.3)  Are appropriate mathematical symbols used, and are all symbols defined? Is the mathematical notation consistent throughout the paper?
    Yes, the paper uses appropriate mathematical symbols and notation consistently throughout.

2.4) Is the paper free from spelling and grammatical errors?
    As far as I can see, the paper has no spelling or grammatical errors.

2.5) Are the figures clear, large enough, and useful?
    Yes.

3. Technical correctness and quality of results
    The technical correctness of the paper is good. The proposed method is effective and efficient for 360 degree multi-task learning. The results are accurate and show good generalization ability. The proposed method is a significant contribution to the field of 360 degree multi-task learning.

4. Adequate reference to previous work (yes/no) ? Please provide reference
     should be cited.
    yes

5. Explanation of recommendation.
   (Detailed comments on the paper and suggestions for improving the paper)
   In the abstract and introduction, there are some things that confuse me. The abstract mentioned that the multi-scale spherical decoder is one of the innovations while the summarized contributions do not include it. Is the multi-scale spherical decoder a new idea or just an improvement over previous work? 
   Besides, in section 3.2, it's hard for the reader to understand self-attention mechanism. It seems that the author assumes the reader knows the details of PanoFormer, but not all readers have the knowledge of PanoFormer. It's better to provide more details about the self-attention mechanism.
   In the ablation study, 'Ours+FB+Fusion' is worse than 'Baseline+FB', what are the differences between them and why it becomes worse after adding fusion module? In addition, 'Ours(final)' only has little improvement compared to 'Ours+FB+Multi-scale', it seems that the fusion module is not as effective as expected. I wonder if sharing the encoder between depth and surface normal estimation could achieve similar results. Or can you use the cross domain attention mechanism proposed in Wonder3D to bridge the gap between depth and surface normal estimation in decoder part?

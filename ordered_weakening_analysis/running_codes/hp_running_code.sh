#!/bin/bash

echo "Running exp 5 ordered_weakening_analysis/head_perturbation_analysis.py"

echo "Running code block 0"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a cubist style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a cubist style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a cubist style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a cubist style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 1"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a cubist style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a pop art style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a pop art style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a pop art style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 2"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a pop art style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a pop art style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a steampunk style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a steampunk style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 3"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a steampunk style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a steampunk style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a steampunk style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a impressionist style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 4"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a impressionist style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a impressionist style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a impressionist style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a impressionist style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 5"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a black-and-white style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a black-and-white style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a black-and-white style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a black-and-white style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 6"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a black-and-white style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a watercolor style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a watercolor style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a watercolor style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 7"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a watercolor style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a watercolor style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a cartoon style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a cartoon style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 8"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a cartoon style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a cartoon style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a cartoon style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a minimalist style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 9"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a minimalist style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a minimalist style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a minimalist style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a minimalist style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 10"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a sepia style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a sepia style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a sepia style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a sepia style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 11"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a sepia style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a sketch style painting of a castle" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 2 --prompt "a sketch style painting of a mountain" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 3 --prompt "a sketch style painting of a cityscape" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait

echo "Running code block 12"
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 0 --prompt "a sketch style painting of a farmland" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&
python ordered_weakening_analysis/head_perturbation_analysis.py --gpu_num 1 --prompt "a sketch style painting of a forest" --concept "Image Style" --experiment_num 5 --seeds 10 20 30&

wait
echo 'Successfully ran the code'

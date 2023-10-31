# python quantization.py --coef_est_type naive --exp_name vit_quant_naive
# python quantization.py --coef_est_type weiner --exp_name quant_attn_ffn_weiner_full
# python quantization.py --coef_est_type weiner --exp_name quant_ffn_weiner_full
# python quantization.py --coef_est_type weiner --exp_name quant_attn_ffn_head_weiner_full
# python quantization.py --coef_est_type weiner --exp_name quant_attn_ffn_head_pembed_weiner_full
# python quantization.py --coef_est_type weiner --exp_name quant_attn_ffnT_head_pembed_weiner_full
# python quantization.py --coef_est_type weiner --exp_name quant_attn_ffn_head_pembed_NoBias_weiner_full
# python quantization.py --coef_est_type weiner --exp_name quant_attn_head_pembed_weiner_full
# python quantization.py --coef_est_type weiner --exp_name quant_attn_ffn_head_pembed_weiner_approx
# python quantization.py --coef_est_type weiner --exp_name quant_attn_head_pembed_weiner_approx
# python quantization.py --coef_est_type weiner --exp_name quant_attn_ffn_head_pembed_weiner_diag
python quantization.py --coef_est_type weiner --exp_name quant_attn_head_pembed_weiner_diag
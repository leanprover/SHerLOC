"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xbf16>
    %4 = "stablehlo.reduce_precision"(%2) <{exponent_bits = 11 : i32, mantissa_bits = 52 : i32}> : (tensor<5x7xbf16>) -> tensor<5x7xbf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x7xbf16>, tensor<5x7xbf16>) -> ()
    "func.return"(%4) : (tensor<5x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3.406250e+00, 3.093750e+00, -5.375000e+00, -2.734380e+00, 1.191410e-01, 2.375000e+00, 7.531250e+00], [-6.875000e+00, 3.000000e+00, -3.015630e+00, -1.828130e+00, -7.187500e-01, -2.624510e-02, 3.007810e-01], [-4.687500e+00, 1.953130e+00, 1.046880e+00, 1.671880e+00, -4.277340e-01, -3.062500e+00, -6.250000e+00], [3.750000e+00, -2.000000e+00, 1.726560e+00, -5.625000e+00, 1.289060e+00, -4.531250e+00, 2.984380e+00], [3.242190e-01, -1.218750e+00, 2.575680e-02, -3.222660e-01, 6.562500e+00, 3.312500e+00, 1.945310e+00]]> : tensor<5x7xbf16>}> : () -> tensor<5x7xbf16>
    "func.return"(%1) : (tensor<5x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-3.406250e+00, 3.093750e+00, -5.375000e+00, -2.734380e+00, 1.191410e-01, 2.375000e+00, 7.531250e+00], [-6.875000e+00, 3.000000e+00, -3.015630e+00, -1.828130e+00, -7.187500e-01, -2.624510e-02, 3.007810e-01], [-4.687500e+00, 1.953130e+00, 1.046880e+00, 1.671880e+00, -4.277340e-01, -3.062500e+00, -6.250000e+00], [3.750000e+00, -2.000000e+00, 1.726560e+00, -5.625000e+00, 1.289060e+00, -4.531250e+00, 2.984380e+00], [3.242190e-01, -1.218750e+00, 2.575680e-02, -3.222660e-01, 6.562500e+00, 3.312500e+00, 1.945310e+00]]> : tensor<5x7xbf16>}> : () -> tensor<5x7xbf16>
    "func.return"(%0) : (tensor<5x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


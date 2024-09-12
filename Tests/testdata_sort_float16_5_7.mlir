"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xf16>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xf16>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
      %6 = "stablehlo.compare"(%arg0, %5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
      %8 = "stablehlo.select"(%6, %7, %arg0) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
      %9 = "stablehlo.compare"(%arg0, %arg0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %10 = "stablehlo.constant"() <{value = dense<0x7E00> : tensor<f16>}> : () -> tensor<f16>
      %11 = "stablehlo.select"(%9, %10, %8) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
      %12 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
      %13 = "stablehlo.compare"(%arg1, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %14 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f16>}> : () -> tensor<f16>
      %15 = "stablehlo.select"(%13, %14, %arg1) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
      %16 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %17 = "stablehlo.constant"() <{value = dense<0x7E00> : tensor<f16>}> : () -> tensor<f16>
      %18 = "stablehlo.select"(%16, %17, %15) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
      %19 = "stablehlo.compare"(%11, %18) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      "stablehlo.return"(%19) : (tensor<i1>) -> ()
    }) : (tensor<5x7xf16>) -> tensor<5x7xf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xf16>, tensor<5x7xf16>) -> ()
    "func.return"(%4) : (tensor<5x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2.148440e+00, 4.452510e-02, 6.804680e+00, 1.814450e+00, 3.199220e+00, -1.463870e+00, 4.503910e+00], [-4.371090e+00, -1.822270e+00, -1.518550e-01, 4.132810e+00, -3.310550e+00, -8.613280e-01, 1.857420e+00], [-2.072270e+00, -2.316410e+00, -3.261720e-01, -9.545890e-01, -5.367190e+00, 1.135740e+00, 3.908200e+00], [-5.170900e-01, -1.863280e+00, 1.348630e+00, -2.921880e+00, 8.007810e-01, 3.398440e+00, -1.192380e+00], [2.033200e+00, -4.953130e+00, 1.804690e+00, -6.308590e-01, -3.638670e+00, -2.794920e+00, -6.117190e+00]]> : tensor<5x7xf16>}> : () -> tensor<5x7xf16>
    "func.return"(%1) : (tensor<5x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.371090e+00, -4.953130e+00, -3.261720e-01, -2.921880e+00, -5.367190e+00, -2.794920e+00, -6.117190e+00], [-2.072270e+00, -2.316410e+00, -1.518550e-01, -9.545890e-01, -3.638670e+00, -1.463870e+00, -1.192380e+00], [-5.170900e-01, -1.863280e+00, 1.348630e+00, -6.308590e-01, -3.310550e+00, -8.613280e-01, 1.857420e+00], [2.033200e+00, -1.822270e+00, 1.804690e+00, 1.814450e+00, 8.007810e-01, 1.135740e+00, 3.908200e+00], [2.148440e+00, 4.452510e-02, 6.804680e+00, 4.132810e+00, 3.199220e+00, 3.398440e+00, 4.503910e+00]]> : tensor<5x7xf16>}> : () -> tensor<5x7xf16>
    "func.return"(%0) : (tensor<5x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


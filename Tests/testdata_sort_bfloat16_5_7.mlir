"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x7xbf16>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5x7xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<5x7xbf16>
    %4 = "stablehlo.sort"(%2) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
      %6 = "stablehlo.compare"(%arg0, %5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
      %8 = "stablehlo.select"(%6, %7, %arg0) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %9 = "stablehlo.compare"(%arg0, %arg0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %10 = "stablehlo.constant"() <{value = dense<0x7FC0> : tensor<bf16>}> : () -> tensor<bf16>
      %11 = "stablehlo.select"(%9, %10, %8) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %12 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
      %13 = "stablehlo.compare"(%arg1, %12) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %14 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
      %15 = "stablehlo.select"(%13, %14, %arg1) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %16 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %17 = "stablehlo.constant"() <{value = dense<0x7FC0> : tensor<bf16>}> : () -> tensor<bf16>
      %18 = "stablehlo.select"(%16, %17, %15) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %19 = "stablehlo.compare"(%11, %18) <{compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      "stablehlo.return"(%19) : (tensor<i1>) -> ()
    }) : (tensor<5x7xbf16>) -> tensor<5x7xbf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7xbf16>, tensor<5x7xbf16>) -> ()
    "func.return"(%4) : (tensor<5x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2.406250e+00, 3.421880e+00, 3.593750e+00, -3.750000e-01, -4.174800e-02, 1.982420e-01, 6.591800e-02], [-6.152340e-02, 1.031250e+00, -7.695310e-01, -8.945310e-01, 7.695310e-01, 2.203130e+00, -2.468750e+00], [4.343750e+00, -4.125000e+00, 3.140630e+00, -9.882810e-01, -1.070310e+00, -5.937500e-01, -3.750000e-01], [-1.240230e-01, -3.046880e+00, 1.937500e+00, -1.460940e+00, -6.884770e-02, -2.687500e+00, -3.468750e+00], [-4.156250e+00, -6.796880e-01, -4.312500e+00, 3.515630e-01, -6.312500e+00, -5.195310e-01, 1.531250e+00]]> : tensor<5x7xbf16>}> : () -> tensor<5x7xbf16>
    "func.return"(%1) : (tensor<5x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.156250e+00, -4.125000e+00, -4.312500e+00, -1.460940e+00, -6.312500e+00, -2.687500e+00, -3.468750e+00], [-1.240230e-01, -3.046880e+00, -7.695310e-01, -9.882810e-01, -1.070310e+00, -5.937500e-01, -2.468750e+00], [-6.152340e-02, -6.796880e-01, 1.937500e+00, -8.945310e-01, -6.884770e-02, -5.195310e-01, -3.750000e-01], [2.406250e+00, 1.031250e+00, 3.140630e+00, -3.750000e-01, -4.174800e-02, 1.982420e-01, 6.591800e-02], [4.343750e+00, 3.421880e+00, 3.593750e+00, 3.515630e-01, 7.695310e-01, 2.203130e+00, 1.531250e+00]]> : tensor<5x7xbf16>}> : () -> tensor<5x7xbf16>
    "func.return"(%0) : (tensor<5x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


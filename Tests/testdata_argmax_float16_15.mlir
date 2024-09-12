"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %15 = "func.call"() <{callee = @inputs}> : () -> tensor<15xf16>
    %16 = "func.call"() <{callee = @expected}> : () -> tensor<i32>
    %17 = "func.call"(%15) <{callee = @argmax}> : (tensor<15xf16>) -> tensor<i32>
    "stablehlo.custom_call"(%17, %16) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i32>, tensor<i32>) -> ()
    "func.return"(%17) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<15xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %14 = "stablehlo.constant"() <{value = dense<[3.396480e+00, 5.957030e+00, -4.455570e-01, -1.531250e+00, 7.988280e-01, -4.769530e+00, -4.462890e-01, 1.254880e+00, -1.915280e-01, -5.424800e-01, 1.443360e+00, 2.171630e-01, -3.675780e+00, -1.687500e+00, 9.174800e-01]> : tensor<15xf16>}> : () -> tensor<15xf16>
    "func.return"(%14) : (tensor<15xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    "func.return"(%13) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<15xf16>) -> tensor<i32>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<15xf16>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<15xi32>
    %1 = "stablehlo.constant"() <{value = dense<0xFC00> : tensor<f16>}> : () -> tensor<f16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<i32>, %arg3: tensor<f16>, %arg4: tensor<i32>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f16>, tensor<f16>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<f16>, tensor<f16>) -> tensor<f16>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%11, %12) : (tensor<f16>, tensor<i32>) -> ()
    }) : (tensor<15xf16>, tensor<15xi32>, tensor<f16>, tensor<i32>) -> (tensor<f16>, tensor<i32>)
    "func.return"(%3#1) : (tensor<i32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


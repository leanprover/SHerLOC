"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %15 = "func.call"() <{callee = @inputs}> : () -> tensor<15xbf16>
    %16 = "func.call"() <{callee = @expected}> : () -> tensor<i32>
    %17 = "func.call"(%15) <{callee = @argmax}> : (tensor<15xbf16>) -> tensor<i32>
    "stablehlo.custom_call"(%17, %16) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i32>, tensor<i32>) -> ()
    "func.return"(%17) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<15xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %14 = "stablehlo.constant"() <{value = dense<[-2.929690e-01, -4.250000e+00, -3.468750e+00, 5.468750e+00, 1.281250e+00, 4.937500e+00, -3.300780e-01, -6.328130e-01, 2.437500e+00, -1.728520e-01, 2.859380e+00, -4.281250e+00, 4.160160e-01, -3.984380e-01, 3.265630e+00]> : tensor<15xbf16>}> : () -> tensor<15xbf16>
    "func.return"(%14) : (tensor<15xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %13 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    "func.return"(%13) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<15xbf16>) -> tensor<i32>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<15xbf16>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<15xi32>
    %1 = "stablehlo.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<i32>, %arg3: tensor<bf16>, %arg4: tensor<i32>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%11, %12) : (tensor<bf16>, tensor<i32>) -> ()
    }) : (tensor<15xbf16>, tensor<15xi32>, tensor<bf16>, tensor<i32>) -> (tensor<bf16>, tensor<i32>)
    "func.return"(%3#1) : (tensor<i32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


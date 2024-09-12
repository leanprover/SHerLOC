"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %15 = "func.call"() <{callee = @inputs}> : () -> tensor<15xui16>
    %16 = "func.call"() <{callee = @expected}> : () -> tensor<i32>
    %17 = "func.call"(%15) <{callee = @argmin}> : (tensor<15xui16>) -> tensor<i32>
    "stablehlo.custom_call"(%17, %16) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i32>, tensor<i32>) -> ()
    "func.return"(%17) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<15xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %14 = "stablehlo.constant"() <{value = dense<[4, 1, 0, 2, 2, 2, 0, 3, 0, 4, 2, 1, 0, 3, 3]> : tensor<15xui16>}> : () -> tensor<15xui16>
    "func.return"(%14) : (tensor<15xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %13 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    "func.return"(%13) : (tensor<i32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<15xui16>) -> tensor<i32>, sym_name = "argmin", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<15xui16>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<15xi32>
    %1 = "stablehlo.constant"() <{value = dense<65535> : tensor<ui16>}> : () -> tensor<ui16>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg1: tensor<ui16>, %arg2: tensor<i32>, %arg3: tensor<ui16>, %arg4: tensor<i32>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%11, %12) : (tensor<ui16>, tensor<i32>) -> ()
    }) : (tensor<15xui16>, tensor<15xi32>, tensor<ui16>, tensor<i32>) -> (tensor<ui16>, tensor<i32>)
    "func.return"(%3#1) : (tensor<i32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


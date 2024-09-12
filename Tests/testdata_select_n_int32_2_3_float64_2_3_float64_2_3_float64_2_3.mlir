"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi32>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf64>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %9 = "stablehlo.compare"(%5#0, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %13 = "stablehlo.select"(%12, %5#2, %5#3) : (tensor<2x3xi1>, tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    %14 = "stablehlo.select"(%9, %5#1, %13) : (tensor<2x3xi1>, tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    "stablehlo.custom_call"(%14, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    "func.return"(%14) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi32>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, 1, 2], [1, 0, 0]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[-1.052776689176447, 4.8069485917233958, -3.7651505536904235], [-2.876392999881408, -1.7670196129089768, -3.6395192156358052]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    %3 = "stablehlo.constant"() <{value = dense<[[2.2510729191479317, 3.1269453789201296, -2.8835289165705293], [-3.3451178542512645, -2.4054327548529431, -4.0062719301466148]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    %4 = "stablehlo.constant"() <{value = dense<[[-0.032373297594009726, -0.90084348243514655, 0.75337834447118979], [0.48588967938531824, -4.7814668477506181, 3.2187831352023175]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1, %2, %3, %4) : (tensor<2x3xi32>, tensor<2x3xf64>, tensor<2x3xf64>, tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-0.032373297594009726, 3.1269453789201296, 0.75337834447118979], [-3.3451178542512645, -1.7670196129089768, -3.6395192156358052]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%0) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


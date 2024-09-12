"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi8>, tensor<3x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi8>) -> tensor<4x3xf64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf64>) -> tensor<3x6xf64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    "func.return"(%7) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi8>, tensor<3x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[5, 3, -1], [0, 2, 1], [1, 2, 1], [-1, 4, 4]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0.1880662032510263, -2.023795304652626, 3.2473837819141886, 4.519621709716283, 4.1380737688287912, 5.7340414107154203], [1.0822822150164857, -1.3397161028972739, -4.1601253183831091, -4.1171799221794778, 1.6826484553523873, 1.481017467936709], [-1.6727293935913132, 0.37300986215698184, 2.1755047526098017, -1.7093730339521322, -0.91975375624215205, 2.2826186668177404]]> : tensor<3x6xf64>}> : () -> tensor<3x6xf64>
    "func.return"(%1, %2) : (tensor<4x3xi8>, tensor<3x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[5.8599070548959009, -14.511134694111934, 1.5810382018118134, 11.955941815995114, 26.658067966443269, 30.83064079056949], [0.49183503644165816, -2.3064223436375659, -6.144745884156416, -9.9437328783110885, 2.4455431544626225, 5.2446536026911588], [0.67990123969268446, -4.3302176482901924, -2.8973621022422278, -5.4241111685948047, 6.5836169232914141, 10.978695013406579], [-2.549854917550336, -1.8430296583085424, -11.185866045007417, -27.825833534242722, -1.0864949723878503, 9.3205031283023771]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%0) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


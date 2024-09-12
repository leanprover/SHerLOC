"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x6xf64>, tensor<4x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf64>
    %5 = "stablehlo.constant"() <{value = dense<0x7FF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %6 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %7:2 = "stablehlo.reduce_window"(%3#1, %3#0, %5, %6) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>):
      %8 = "stablehlo.compare"(%arg0, %arg2) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %9 = "stablehlo.select"(%8, %arg0, %arg2) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %10 = "stablehlo.select"(%8, %arg1, %arg3) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%9, %10) : (tensor<f64>, tensor<f64>) -> ()
    }) : (tensor<4x6xf64>, tensor<4x6xf64>, tensor<f64>, tensor<f64>) -> (tensor<3x5xf64>, tensor<3x5xf64>)
    "stablehlo.custom_call"(%7#1, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf64>, tensor<3x5xf64>) -> ()
    "func.return"(%7#1) : (tensor<3x5xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x6xf64>, tensor<4x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-7.1149372968688649, 0.06320878551452172, -2.7274963731057098, 2.2385959323403424, 0.1852081754565747, -0.34050181539936386], [1.218581858737378, -1.8100311668811839, -2.0416048520467962, 3.3229527312744809, -1.5791817244234267, 0.41975500990071585], [-0.39454487826059792, 2.4831068786241488, -0.59727638631645219, -3.1629397784485924, 2.7649750466923506, 6.4629911464084318], [-1.3053018707290454, -1.0661663537704877, -1.9961123005977237, 2.057529159538126, -1.2635029031873009, 3.5539599255940049]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[2.6871646805053282, 1.1079280583580291, 0.071354124721101739, 2.366803437360061, -0.24529864383761962, -3.5082838238461784], [0.2256227690737253, -2.9742316811601768, 2.5802444178171275, 1.624968440403348, -1.332422678704781, 2.9098649030965662], [1.861052554504345, 0.75490544268091297, 4.0689720446555757, -2.51724204911234, -0.41140104575078318, 1.0564941083029569], [4.4932466380783289, -2.5840047455330049, -2.3001326558887865, 3.9169581915115663, 0.17115619727584172, -1.8850388829842393]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%1, %2) : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1.8100311668811839, -1.8100311668811839, -2.7274963731057098, -1.5791817244234267, -0.34050181539936386], [-1.8100311668811839, -1.8100311668811839, -3.1629397784485924, -3.1629397784485924, -1.5791817244234267], [-1.0661663537704877, -1.0661663537704877, -3.1629397784485924, -3.1629397784485924, 3.5539599255940049]]> : tensor<3x5xf64>}> : () -> tensor<3x5xf64>
    "func.return"(%0) : (tensor<3x5xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


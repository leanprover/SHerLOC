"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x1x6xf32>, tensor<2x4x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xf32>
    %5 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%10) : (tensor<f32>) -> ()
    }) : (tensor<2x4x6xf32>, tensor<2x1x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    "func.return"(%9) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x1x6xf32>, tensor<2x4x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[1.19308376, 0.574337304, 0.771823287, -0.544335902, -3.51993728, 2.39790201]], [[-1.84060347, 2.64280605, 1.53013074, -2.40419912, 3.80637503, 1.55404234]]]> : tensor<2x1x6xf32>}> : () -> tensor<2x1x6xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[3.8620379, 2.20822835, 3.35131288, -1.51076293, 1.85093927, -3.03109074], [1.26185918, 6.36994696, 1.05354285, -2.17369485, 3.45685482, -6.95882701], [6.67202854, 0.383125722, 0.0620134398, 1.42351806, -2.77465606, 2.41579247], [2.30130601, 3.26142049, 0.825048804, 1.56609392, -2.18073368, 2.59537792]], [[1.84301186, -6.02817058, 1.54950094, 4.0299511, -0.491929442, -8.70993995], [3.2347393, 3.72320461, -2.91740704, 1.03011239, 5.61139584, -3.50814581], [3.689810e-01, -0.252711564, 0.726274729, 0.11600668, -0.334616214, -4.00004578], [-2.13016582, -0.317696303, 1.65582919, 6.62267398, 1.85173309, -0.621818244]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%1, %2) : (tensor<2x1x6xf32>, tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, 0.000000e+00, 0.771823287, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.574337304, 0.000000e+00, 0.000000e+00, -3.51993728, 0.000000e+00], [1.19308376, 0.000000e+00, 0.000000e+00, -0.544335902, 0.000000e+00, 2.39790201], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 1.53013074, -2.40419912, 0.000000e+00, 0.000000e+00], [-1.84060347, 2.64280605, 0.000000e+00, 0.000000e+00, 3.80637503, 1.55404234], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%0) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x3x5xf64>, tensor<2x4x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xf64>
    %5 = "stablehlo.constant"() <{value = dense<0xFFF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xf64>, tensor<f64>) -> tensor<2x4x6xf64>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg2: tensor<f64>, %arg3: tensor<f64>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%10) : (tensor<f64>) -> ()
    }) : (tensor<2x4x6xf64>, tensor<1x3x5xf64>, tensor<f64>) -> tensor<2x4x6xf64>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xf64>) -> tensor<2x4x6xf64>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xf64>, tensor<2x4x6xf64>) -> ()
    "func.return"(%9) : (tensor<2x4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x3x5xf64>, tensor<2x4x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[2.7169098713534892, -0.55152663672318247, 1.1695045284769516, -0.88900009421875981, 0.24097687554049244], [-2.5360819282048714, 4.415075956580286, -3.0972122743594817, -1.6937196918523394, -0.44042857486494036], [1.3921436691148026, -0.85015751035636588, -0.46025290531372842, -2.771999678067437, -5.9841490617042368]]]> : tensor<1x3x5xf64>}> : () -> tensor<1x3x5xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[[0.37238723524483464, 8.1931106234492912, 0.22586763421781164, 5.0921318206743296, -1.8200476972887705, 2.0139155702084315], [-3.0758886214443173, -1.7082632340002804, -0.94514195495943642, -1.1259585493474531, -1.7955406628958794, 4.8580726235699476], [2.6379028821462667, 2.3565664186939395, -4.9281398721070069, 3.0522098899990882, -0.19919843678308524, -2.5044764485288322], [3.2229548027032866, 1.3881457611825061, 0.81334724494064115, -0.6214095635926522, 1.4429580951941059, 0.10499413855969078]], [[-4.5184953881864027, 3.5254899997798237, 1.39590524565681, -0.15917820745188258, 0.3183127450989639, -2.7166923404860146], [2.536031222255521, -3.6991989870635296, 0.011261675253965165, -4.8074662424777088, 2.4336670754210243, 1.4661329589698144], [2.4423392106546697, 5.4522494734588252, 2.8199715066846425, 2.6956157617430336, -2.3799176897652954, -3.8043731449309623], [-1.0980479414827662, 0.9046470467541714, 1.845449014745268, 0.7099613481172331, -2.5025262885713784, 3.6280404310317023]]]> : tensor<2x4x6xf64>}> : () -> tensor<2x4x6xf64>
    "func.return"(%1, %2) : (tensor<1x3x5xf64>, tensor<2x4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, 2.1653832346303066, 0.000000e+00, 0.28050443425819183, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -0.19945169932444792], [0.000000e+00, 0.000000e+00, 0.000000e+00, -8.0231845495929868, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 2.4209801871338512, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -5.9841490617042368]]]> : tensor<2x4x6xf64>}> : () -> tensor<2x4x6xf64>
    "func.return"(%0) : (tensor<2x4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


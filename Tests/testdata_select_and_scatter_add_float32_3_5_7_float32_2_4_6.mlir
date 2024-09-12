"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x7xf32>, tensor<2x4x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xf32>
    %5 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 1, 1, 1>, edge_padding_low = array<i64: 1, 1, 1>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<4x6x8xf32>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%10) : (tensor<f32>) -> ()
    }) : (tensor<4x6x8xf32>, tensor<3x5x7xf32>, tensor<f32>) -> tensor<4x6x8xf32>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 3, 5, 7>, start_indices = array<i64: 1, 1, 1>, strides = array<i64: 1, 1, 1>}> : (tensor<4x6x8xf32>) -> tensor<2x4x6xf32>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> ()
    "func.return"(%9) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x7xf32>, tensor<2x4x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x14BA8140EE7C8C40B72358BF44F2E4BF768518BEFDA53C40AD568A3F3AF382C0DA30B23FA30A4DBE19FA58406E2AA74061EF30C0060415C0C09C093F8ED504C18517CB3FEA12E0400A5C86C08ECE883F0A351F40784FB1409D6C8EC04119DABFC2628840E683B3C0E4CC7FC08D6F32402114844040901D4097D415BF3C7C1BC0C67094BF4F06574002358340AD4DA83FCAD2E43F1FA72CC0D739C9BE5229A1BF658E34BFB0CAF4BD184EC13F3EF3F1BF0B17DAC092412E403192934065E454C0FEAC61C063E230402A4929BF6D831ABE2E819F40F99E71BFFB91E93FCD43B5BF22B39240FF729CBFAFA8174055165CC0DEC273406E969F4067F6C1BF9AD4633FA240603F6EC866BFD8AFF8BF3906633E2E7A08405F6D7B3F1B2933BCE0971840D74E42407C9A2C3FD193BC3F8B51D7BEB7905EBF5FE08740622732C051FE1B4013D5A840B1C09DBEED4DCDBEA58C14BF4A1E20C0D75E073FA9E378C098635240D9D3E63FBBEABCBFEB839EC052D8B7C0836C3940DDF29CC0BCD0F23F4C9BC63FE211A73FA55E2E3FE3082E4033559340ADCF07C08751AD40003F2BC074D45FC00FC319C0"> : tensor<3x5x7xf32>}> : () -> tensor<3x5x7xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[9.620100e+00, -2.12728262, -1.56567204, 3.59518933, 1.462800e+00, -4.86497736], [1.74625087, -0.00808556098, -0.0342763178, 2.07804894, -0.964987456, -5.13031816], [0.0146070309, -1.24919343, -5.25079107, 1.74715388, 0.203711271, 0.950955867], [0.189528704, -1.79846704, -4.15770769, -3.86397576, 3.24413562, 1.84910381]], [[1.46117151, -4.5444808, -1.31956887, -0.0305220447, 2.35218406, -3.15633678], [-1.86811352, 1.90593982, 0.378701568, -1.22142124, -1.6682936, 5.19656277], [1.08109593, 2.81859517, -3.05214119, 2.40367198, 1.07703853, 2.27809072], [-2.89373779, 0.616681277, 0.84218657, 2.28567815, 4.25000191, 3.33190656]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%1, %2) : (tensor<3x5x7xf32>, tensor<2x4x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[8.46665287, 0.000000e+00, -0.844294965, 12.3588591, 0.183020592, -1.24760044], [-5.00077152, 1.38641942, 0.000000e+00, 2.80357361, 0.000000e+00, 0.000000e+00], [0.000000e+00, -1.70389569, 0.000000e+00, 4.2620554, 0.000000e+00, 3.5564158], [8.56954193, -0.585275114, 0.000000e+00, -2.42945766, -7.40666294, 6.88827896]], [[6.61947345, 0.000000e+00, 0.33836174, 0.674232244, 0.0393084884, -0.988922894], [0.000000e+00, -7.1615696, 5.276010e+00, 0.000000e+00, 0.000000e+00, -13.8547611], [-3.66263819, -5.03225613, 0.000000e+00, 7.58948802, 0.000000e+00, 0.000000e+00], [2.7192924, 5.48013639, -3.02354622, 3.47333574, 7.83382701, -2.25459337]]]> : tensor<2x4x6xf32>}> : () -> tensor<2x4x6xf32>
    "func.return"(%0) : (tensor<2x4x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


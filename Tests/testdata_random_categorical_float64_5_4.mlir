"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %131 = "func.call"() <{callee = @inputs}> : () -> tensor<5x4xf64>
    %132 = "func.call"() <{callee = @expected}> : () -> tensor<5xi64>
    %133 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %134 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %135 = "stablehlo.shift_right_logical"(%133, %134) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %136 = "stablehlo.convert"(%135) : (tensor<i64>) -> tensor<ui32>
    %137 = "stablehlo.broadcast_in_dim"(%136) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %138 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %139 = "stablehlo.convert"(%138) : (tensor<ui32>) -> tensor<i64>
    %140 = "stablehlo.and"(%133, %139) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %141 = "stablehlo.convert"(%140) : (tensor<i64>) -> tensor<ui32>
    %142 = "stablehlo.broadcast_in_dim"(%141) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %143 = "stablehlo.concatenate"(%137, %142) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %144 = "func.call"(%143) <{callee = @_gumbel}> : (tensor<2xui32>) -> tensor<5x4xf64>
    %145 = "stablehlo.add"(%144, %131) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    %146 = "func.call"(%145) <{callee = @argmax}> : (tensor<5x4xf64>) -> tensor<5xi64>
    "stablehlo.custom_call"(%146, %132) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5xi64>, tensor<5xi64>) -> ()
    "func.return"(%146) : (tensor<5xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x4xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %130 = "stablehlo.constant"() <{value = dense<[[2.6044160679950643, 2.9321362344077575, -3.2103477133467098, -1.8187160634972799], [2.1577879203212329, 0.94549340118604897, -1.2300833153827209, 6.037781617043855], [2.6481661417030908, 0.76780148062618547, 0.25212015749028915, -2.0297729471187482], [-8.2722376135920346, 2.0099702230233962, 3.0598081711609653, -1.2354732207389678], [-3.2511804161164988, 1.7202350343151254, 3.8548454509404677, -0.67754275745650916]]> : tensor<5x4xf64>}> : () -> tensor<5x4xf64>
    "func.return"(%130) : (tensor<5x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %129 = "stablehlo.constant"() <{value = dense<[1, 3, 0, 2, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    "func.return"(%129) : (tensor<5xi64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<5x4xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_gumbel", sym_visibility = "private"}> ({
  ^bb0(%arg34: tensor<2xui32>):
    %122 = "stablehlo.constant"() <{value = dense<2.2250738585072014E-308> : tensor<f64>}> : () -> tensor<f64>
    %123 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %124 = "func.call"(%arg34, %122, %123) <{callee = @_uniform}> : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<5x4xf64>
    %125 = "stablehlo.log"(%124) : (tensor<5x4xf64>) -> tensor<5x4xf64>
    %126 = "stablehlo.negate"(%125) : (tensor<5x4xf64>) -> tensor<5x4xf64>
    %127 = "stablehlo.log"(%126) : (tensor<5x4xf64>) -> tensor<5x4xf64>
    %128 = "stablehlo.negate"(%127) : (tensor<5x4xf64>) -> tensor<5x4xf64>
    "func.return"(%128) : (tensor<5x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<5x4xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform", sym_visibility = "private"}> ({
  ^bb0(%arg13: tensor<2xui32>, %arg14: tensor<f64>, %arg15: tensor<f64>):
    %68 = "stablehlo.convert"(%arg15) : (tensor<f64>) -> tensor<f64>
    %69 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1x1xf64>
    %70 = "stablehlo.broadcast_in_dim"(%68) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1x1xf64>
    %71 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<40xui32>
    %72 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %73 = "stablehlo.reshape"(%72) : (tensor<1xui32>) -> tensor<ui32>
    %74 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %75 = "stablehlo.reshape"(%74) : (tensor<1xui32>) -> tensor<ui32>
    %76 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 20>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<40xui32>) -> tensor<20xui32>
    %77 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 40>, start_indices = array<i64: 20>, strides = array<i64: 1>}> : (tensor<40xui32>) -> tensor<20xui32>
    %78 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %79 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %80 = "stablehlo.xor"(%73, %75) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %81 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %82 = "stablehlo.xor"(%80, %81) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %83 = "stablehlo.broadcast_in_dim"(%73) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %84 = "stablehlo.add"(%76, %83) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %85 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %86 = "stablehlo.add"(%77, %85) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %87 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %88 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %89:9 = "stablehlo.while"(%88, %87, %84, %86, %75, %82, %73, %78, %79) ({
    ^bb0(%arg25: tensor<i64>, %arg26: tensor<i64>, %arg27: tensor<20xui32>, %arg28: tensor<20xui32>, %arg29: tensor<ui32>, %arg30: tensor<ui32>, %arg31: tensor<ui32>, %arg32: tensor<4xui32>, %arg33: tensor<4xui32>):
      %120 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %121 = "stablehlo.compare"(%arg25, %120) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%121) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg16: tensor<i64>, %arg17: tensor<i64>, %arg18: tensor<20xui32>, %arg19: tensor<20xui32>, %arg20: tensor<ui32>, %arg21: tensor<ui32>, %arg22: tensor<ui32>, %arg23: tensor<4xui32>, %arg24: tensor<4xui32>):
      %117:8 = "func.call"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24) <{callee = @None}> : (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %118 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %119 = "stablehlo.add"(%arg16, %118) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%119, %117#0, %117#1, %117#2, %117#3, %117#4, %117#5, %117#6, %117#7) : (tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %90 = "stablehlo.concatenate"(%89#2, %89#3) <{dimension = 0 : i64}> : (tensor<20xui32>, tensor<20xui32>) -> tensor<40xui32>
    %91 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 20>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<40xui32>) -> tensor<20xui32>
    %92 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 40>, start_indices = array<i64: 20>, strides = array<i64: 1>}> : (tensor<40xui32>) -> tensor<20xui32>
    %93 = "stablehlo.convert"(%91) : (tensor<20xui32>) -> tensor<20xui64>
    %94 = "stablehlo.convert"(%92) : (tensor<20xui32>) -> tensor<20xui64>
    %95 = "stablehlo.constant"() <{value = dense<32> : tensor<ui64>}> : () -> tensor<ui64>
    %96 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<20xui64>
    %97 = "stablehlo.shift_left"(%93, %96) : (tensor<20xui64>, tensor<20xui64>) -> tensor<20xui64>
    %98 = "stablehlo.or"(%97, %94) : (tensor<20xui64>, tensor<20xui64>) -> tensor<20xui64>
    %99 = "stablehlo.reshape"(%98) : (tensor<20xui64>) -> tensor<5x4xui64>
    %100 = "stablehlo.constant"() <{value = dense<12> : tensor<ui64>}> : () -> tensor<ui64>
    %101 = "stablehlo.broadcast_in_dim"(%100) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<5x4xui64>
    %102 = "stablehlo.shift_right_logical"(%99, %101) : (tensor<5x4xui64>, tensor<5x4xui64>) -> tensor<5x4xui64>
    %103 = "stablehlo.constant"() <{value = dense<4607182418800017408> : tensor<ui64>}> : () -> tensor<ui64>
    %104 = "stablehlo.broadcast_in_dim"(%103) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<5x4xui64>
    %105 = "stablehlo.or"(%102, %104) : (tensor<5x4xui64>, tensor<5x4xui64>) -> tensor<5x4xui64>
    %106 = "stablehlo.bitcast_convert"(%105) : (tensor<5x4xui64>) -> tensor<5x4xf64>
    %107 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %108 = "stablehlo.broadcast_in_dim"(%107) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<5x4xf64>
    %109 = "stablehlo.subtract"(%106, %108) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    %110 = "stablehlo.subtract"(%70, %69) : (tensor<1x1xf64>, tensor<1x1xf64>) -> tensor<1x1xf64>
    %111 = "stablehlo.broadcast_in_dim"(%110) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %112 = "stablehlo.multiply"(%109, %111) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    %113 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %114 = "stablehlo.add"(%112, %113) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    %115 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x1xf64>) -> tensor<5x4xf64>
    %116 = "stablehlo.maximum"(%115, %114) : (tensor<5x4xf64>, tensor<5x4xf64>) -> tensor<5x4xf64>
    "func.return"(%116) : (tensor<5x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg5: tensor<i64>, %arg6: tensor<20xui32>, %arg7: tensor<20xui32>, %arg8: tensor<ui32>, %arg9: tensor<ui32>, %arg10: tensor<ui32>, %arg11: tensor<4xui32>, %arg12: tensor<4xui32>):
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %14 = "stablehlo.add"(%arg5, %13) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %15 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %16 = "stablehlo.reshape"(%15) : (tensor<1xui32>) -> tensor<ui32>
    %17 = "stablehlo.add"(%arg6, %arg7) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %18 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %19 = "stablehlo.shift_left"(%arg7, %18) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %20 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %21 = "stablehlo.subtract"(%20, %16) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %23 = "stablehlo.shift_right_logical"(%arg7, %22) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %24 = "stablehlo.or"(%19, %23) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %25 = "stablehlo.xor"(%17, %24) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %26 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %27 = "stablehlo.reshape"(%26) : (tensor<1xui32>) -> tensor<ui32>
    %28 = "stablehlo.add"(%17, %25) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %29 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %30 = "stablehlo.shift_left"(%25, %29) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %31 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %32 = "stablehlo.subtract"(%31, %27) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %33 = "stablehlo.broadcast_in_dim"(%32) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %34 = "stablehlo.shift_right_logical"(%25, %33) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %35 = "stablehlo.or"(%30, %34) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %36 = "stablehlo.xor"(%28, %35) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %37 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %38 = "stablehlo.reshape"(%37) : (tensor<1xui32>) -> tensor<ui32>
    %39 = "stablehlo.add"(%28, %36) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %40 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %41 = "stablehlo.shift_left"(%36, %40) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %42 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %43 = "stablehlo.subtract"(%42, %38) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %45 = "stablehlo.shift_right_logical"(%36, %44) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %46 = "stablehlo.or"(%41, %45) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %47 = "stablehlo.xor"(%39, %46) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %48 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %49 = "stablehlo.reshape"(%48) : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.add"(%39, %47) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %51 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %52 = "stablehlo.shift_left"(%47, %51) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %53 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %54 = "stablehlo.subtract"(%53, %49) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %56 = "stablehlo.shift_right_logical"(%47, %55) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %57 = "stablehlo.or"(%52, %56) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %58 = "stablehlo.xor"(%50, %57) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %59 = "stablehlo.broadcast_in_dim"(%arg8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %60 = "stablehlo.add"(%50, %59) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %61 = "stablehlo.broadcast_in_dim"(%arg9) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %62 = "stablehlo.add"(%58, %61) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    %63 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %64 = "stablehlo.add"(%arg5, %63) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %65 = "stablehlo.convert"(%64) : (tensor<i64>) -> tensor<ui32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<20xui32>
    %67 = "stablehlo.add"(%62, %66) : (tensor<20xui32>, tensor<20xui32>) -> tensor<20xui32>
    "func.return"(%14, %60, %67, %arg9, %arg10, %arg8, %arg12, %arg11) : (tensor<i64>, tensor<20xui32>, tensor<20xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<5x4xf64>) -> tensor<5xi64>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<5x4xf64>):
    %0 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<5x4xi64>
    %1 = "stablehlo.constant"() <{value = dense<0xFFF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg1: tensor<f64>, %arg2: tensor<i64>, %arg3: tensor<f64>, %arg4: tensor<i64>):
      %4 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %5 = "stablehlo.compare"(%arg1, %arg1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %6 = "stablehlo.or"(%4, %5) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %7 = "stablehlo.compare"(%arg1, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %8 = "stablehlo.compare"(%arg2, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %9 = "stablehlo.and"(%7, %8) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %10 = "stablehlo.or"(%6, %9) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      %11 = "stablehlo.select"(%6, %arg1, %arg3) : (tensor<i1>, tensor<f64>, tensor<f64>) -> tensor<f64>
      %12 = "stablehlo.select"(%10, %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%11, %12) : (tensor<f64>, tensor<i64>) -> ()
    }) : (tensor<5x4xf64>, tensor<5x4xi64>, tensor<f64>, tensor<i64>) -> (tensor<5xf64>, tensor<5xi64>)
    "func.return"(%3#1) : (tensor<5xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


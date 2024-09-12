"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<i64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %130 = "func.call"() <{callee = @inputs}> : () -> tensor<8xf64>
    %131 = "func.call"() <{callee = @expected}> : () -> tensor<i64>
    %132 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %133 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %134 = "stablehlo.shift_right_logical"(%132, %133) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %135 = "stablehlo.convert"(%134) : (tensor<i64>) -> tensor<ui32>
    %136 = "stablehlo.broadcast_in_dim"(%135) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %137 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %138 = "stablehlo.convert"(%137) : (tensor<ui32>) -> tensor<i64>
    %139 = "stablehlo.and"(%132, %138) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %140 = "stablehlo.convert"(%139) : (tensor<i64>) -> tensor<ui32>
    %141 = "stablehlo.broadcast_in_dim"(%140) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %142 = "stablehlo.concatenate"(%136, %141) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %143 = "func.call"(%142) <{callee = @_gumbel}> : (tensor<2xui32>) -> tensor<8xf64>
    %144 = "stablehlo.add"(%143, %130) : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xf64>
    %145 = "func.call"(%144) <{callee = @argmax}> : (tensor<8xf64>) -> tensor<i64>
    "stablehlo.custom_call"(%145, %131) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<i64>, tensor<i64>) -> ()
    "func.return"(%145) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %129 = "stablehlo.constant"() <{value = dense<[-7.1183656526382366, -0.96955133078721589, -1.6736512891996449, 7.2285932197100164, -7.3226303506590709, -0.12728901223196259, 2.9544739110238858, -3.466593325644622]> : tensor<8xf64>}> : () -> tensor<8xf64>
    "func.return"(%129) : (tensor<8xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<i64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %128 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    "func.return"(%128) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<8xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_gumbel", sym_visibility = "private"}> ({
  ^bb0(%arg34: tensor<2xui32>):
    %121 = "stablehlo.constant"() <{value = dense<2.2250738585072014E-308> : tensor<f64>}> : () -> tensor<f64>
    %122 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %123 = "func.call"(%arg34, %121, %122) <{callee = @_uniform}> : (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<8xf64>
    %124 = "stablehlo.log"(%123) : (tensor<8xf64>) -> tensor<8xf64>
    %125 = "stablehlo.negate"(%124) : (tensor<8xf64>) -> tensor<8xf64>
    %126 = "stablehlo.log"(%125) : (tensor<8xf64>) -> tensor<8xf64>
    %127 = "stablehlo.negate"(%126) : (tensor<8xf64>) -> tensor<8xf64>
    "func.return"(%127) : (tensor<8xf64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<f64>, tensor<f64>) -> tensor<8xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_uniform", sym_visibility = "private"}> ({
  ^bb0(%arg13: tensor<2xui32>, %arg14: tensor<f64>, %arg15: tensor<f64>):
    %68 = "stablehlo.convert"(%arg15) : (tensor<f64>) -> tensor<f64>
    %69 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %70 = "stablehlo.broadcast_in_dim"(%68) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<1xf64>
    %71 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<16xui32>
    %72 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %73 = "stablehlo.reshape"(%72) : (tensor<1xui32>) -> tensor<ui32>
    %74 = "stablehlo.slice"(%arg13) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %75 = "stablehlo.reshape"(%74) : (tensor<1xui32>) -> tensor<ui32>
    %76 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 8>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<16xui32>) -> tensor<8xui32>
    %77 = "stablehlo.slice"(%71) <{limit_indices = array<i64: 16>, start_indices = array<i64: 8>, strides = array<i64: 1>}> : (tensor<16xui32>) -> tensor<8xui32>
    %78 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %79 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %80 = "stablehlo.xor"(%73, %75) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %81 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %82 = "stablehlo.xor"(%80, %81) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %83 = "stablehlo.broadcast_in_dim"(%73) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %84 = "stablehlo.add"(%76, %83) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %85 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %86 = "stablehlo.add"(%77, %85) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %87 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %88 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %89:9 = "stablehlo.while"(%88, %87, %84, %86, %75, %82, %73, %78, %79) ({
    ^bb0(%arg25: tensor<i64>, %arg26: tensor<i64>, %arg27: tensor<8xui32>, %arg28: tensor<8xui32>, %arg29: tensor<ui32>, %arg30: tensor<ui32>, %arg31: tensor<ui32>, %arg32: tensor<4xui32>, %arg33: tensor<4xui32>):
      %119 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %120 = "stablehlo.compare"(%arg25, %119) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%120) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg16: tensor<i64>, %arg17: tensor<i64>, %arg18: tensor<8xui32>, %arg19: tensor<8xui32>, %arg20: tensor<ui32>, %arg21: tensor<ui32>, %arg22: tensor<ui32>, %arg23: tensor<4xui32>, %arg24: tensor<4xui32>):
      %116:8 = "func.call"(%arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24) <{callee = @None}> : (tensor<i64>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %117 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %118 = "stablehlo.add"(%arg16, %117) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%118, %116#0, %116#1, %116#2, %116#3, %116#4, %116#5, %116#6, %116#7) : (tensor<i64>, tensor<i64>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %90 = "stablehlo.concatenate"(%89#2, %89#3) <{dimension = 0 : i64}> : (tensor<8xui32>, tensor<8xui32>) -> tensor<16xui32>
    %91 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 8>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<16xui32>) -> tensor<8xui32>
    %92 = "stablehlo.slice"(%90) <{limit_indices = array<i64: 16>, start_indices = array<i64: 8>, strides = array<i64: 1>}> : (tensor<16xui32>) -> tensor<8xui32>
    %93 = "stablehlo.convert"(%91) : (tensor<8xui32>) -> tensor<8xui64>
    %94 = "stablehlo.convert"(%92) : (tensor<8xui32>) -> tensor<8xui64>
    %95 = "stablehlo.constant"() <{value = dense<32> : tensor<ui64>}> : () -> tensor<ui64>
    %96 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<8xui64>
    %97 = "stablehlo.shift_left"(%93, %96) : (tensor<8xui64>, tensor<8xui64>) -> tensor<8xui64>
    %98 = "stablehlo.or"(%97, %94) : (tensor<8xui64>, tensor<8xui64>) -> tensor<8xui64>
    %99 = "stablehlo.constant"() <{value = dense<12> : tensor<ui64>}> : () -> tensor<ui64>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<8xui64>
    %101 = "stablehlo.shift_right_logical"(%98, %100) : (tensor<8xui64>, tensor<8xui64>) -> tensor<8xui64>
    %102 = "stablehlo.constant"() <{value = dense<4607182418800017408> : tensor<ui64>}> : () -> tensor<ui64>
    %103 = "stablehlo.broadcast_in_dim"(%102) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<8xui64>
    %104 = "stablehlo.or"(%101, %103) : (tensor<8xui64>, tensor<8xui64>) -> tensor<8xui64>
    %105 = "stablehlo.bitcast_convert"(%104) : (tensor<8xui64>) -> tensor<8xf64>
    %106 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %107 = "stablehlo.broadcast_in_dim"(%106) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<8xf64>
    %108 = "stablehlo.subtract"(%105, %107) : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xf64>
    %109 = "stablehlo.subtract"(%70, %69) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %110 = "stablehlo.broadcast_in_dim"(%109) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf64>) -> tensor<8xf64>
    %111 = "stablehlo.multiply"(%108, %110) : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xf64>
    %112 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf64>) -> tensor<8xf64>
    %113 = "stablehlo.add"(%111, %112) : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xf64>
    %114 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xf64>) -> tensor<8xf64>
    %115 = "stablehlo.maximum"(%114, %113) : (tensor<8xf64>, tensor<8xf64>) -> tensor<8xf64>
    "func.return"(%115) : (tensor<8xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg5: tensor<i64>, %arg6: tensor<8xui32>, %arg7: tensor<8xui32>, %arg8: tensor<ui32>, %arg9: tensor<ui32>, %arg10: tensor<ui32>, %arg11: tensor<4xui32>, %arg12: tensor<4xui32>):
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %14 = "stablehlo.add"(%arg5, %13) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %15 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %16 = "stablehlo.reshape"(%15) : (tensor<1xui32>) -> tensor<ui32>
    %17 = "stablehlo.add"(%arg6, %arg7) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %18 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %19 = "stablehlo.shift_left"(%arg7, %18) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %20 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %21 = "stablehlo.subtract"(%20, %16) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %23 = "stablehlo.shift_right_logical"(%arg7, %22) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %24 = "stablehlo.or"(%19, %23) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %25 = "stablehlo.xor"(%17, %24) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %26 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %27 = "stablehlo.reshape"(%26) : (tensor<1xui32>) -> tensor<ui32>
    %28 = "stablehlo.add"(%17, %25) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %29 = "stablehlo.broadcast_in_dim"(%27) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %30 = "stablehlo.shift_left"(%25, %29) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %31 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %32 = "stablehlo.subtract"(%31, %27) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %33 = "stablehlo.broadcast_in_dim"(%32) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %34 = "stablehlo.shift_right_logical"(%25, %33) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %35 = "stablehlo.or"(%30, %34) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %36 = "stablehlo.xor"(%28, %35) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %37 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %38 = "stablehlo.reshape"(%37) : (tensor<1xui32>) -> tensor<ui32>
    %39 = "stablehlo.add"(%28, %36) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %40 = "stablehlo.broadcast_in_dim"(%38) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %41 = "stablehlo.shift_left"(%36, %40) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %42 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %43 = "stablehlo.subtract"(%42, %38) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %44 = "stablehlo.broadcast_in_dim"(%43) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %45 = "stablehlo.shift_right_logical"(%36, %44) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %46 = "stablehlo.or"(%41, %45) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %47 = "stablehlo.xor"(%39, %46) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %48 = "stablehlo.slice"(%arg11) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %49 = "stablehlo.reshape"(%48) : (tensor<1xui32>) -> tensor<ui32>
    %50 = "stablehlo.add"(%39, %47) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %51 = "stablehlo.broadcast_in_dim"(%49) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %52 = "stablehlo.shift_left"(%47, %51) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %53 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %54 = "stablehlo.subtract"(%53, %49) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %55 = "stablehlo.broadcast_in_dim"(%54) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %56 = "stablehlo.shift_right_logical"(%47, %55) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %57 = "stablehlo.or"(%52, %56) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %58 = "stablehlo.xor"(%50, %57) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %59 = "stablehlo.broadcast_in_dim"(%arg8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %60 = "stablehlo.add"(%50, %59) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %61 = "stablehlo.broadcast_in_dim"(%arg9) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %62 = "stablehlo.add"(%58, %61) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    %63 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %64 = "stablehlo.add"(%arg5, %63) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %65 = "stablehlo.convert"(%64) : (tensor<i64>) -> tensor<ui32>
    %66 = "stablehlo.broadcast_in_dim"(%65) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<8xui32>
    %67 = "stablehlo.add"(%62, %66) : (tensor<8xui32>, tensor<8xui32>) -> tensor<8xui32>
    "func.return"(%14, %60, %67, %arg9, %arg10, %arg8, %arg12, %arg11) : (tensor<i64>, tensor<8xui32>, tensor<8xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8xf64>) -> tensor<i64>, sym_name = "argmax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8xf64>):
    %0 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<8xi64>
    %1 = "stablehlo.constant"() <{value = dense<0xFFF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %3:2 = "stablehlo.reduce"(%arg0, %0, %1, %2) <{dimensions = array<i64: 0>}> ({
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
    }) : (tensor<8xf64>, tensor<8xi64>, tensor<f64>, tensor<i64>) -> (tensor<f64>, tensor<i64>)
    "func.return"(%3#1) : (tensor<i64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


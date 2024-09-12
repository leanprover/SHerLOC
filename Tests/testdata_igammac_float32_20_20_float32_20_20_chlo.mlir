"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf32>, tensor<20x20xf32>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %9 = "stablehlo.compare"(%5#1, %8) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %10 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %13 = "stablehlo.or"(%9, %12) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %14 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %15 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %16 = "stablehlo.compare"(%5#1, %15) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %17 = "stablehlo.compare"(%5#1, %5#0) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = "stablehlo.or"(%16, %17) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %19 = "stablehlo.log"(%5#1) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %20 = "stablehlo.multiply"(%5#0, %19) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %21 = "stablehlo.subtract"(%20, %5#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %22 = "chlo.lgamma"(%5#0) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %23 = "stablehlo.subtract"(%21, %22) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %24 = "stablehlo.constant"() <{value = dense<3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
    %25 = "stablehlo.log"(%24) : (tensor<f32>) -> tensor<f32>
    %26 = "stablehlo.negate"(%25) : (tensor<f32>) -> tensor<f32>
    %27 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %28 = "stablehlo.compare"(%23, %27) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %29 = "stablehlo.or"(%13, %28) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %30 = "stablehlo.not"(%29) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %31 = "stablehlo.exponential"(%23) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "stablehlo.and"(%30, %18) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %33 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %34 = "stablehlo.broadcast_in_dim"(%33) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %35 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %36 = "stablehlo.broadcast_in_dim"(%35) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %37 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %38 = "stablehlo.broadcast_in_dim"(%37) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %39 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %40 = "stablehlo.broadcast_in_dim"(%39) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %41:7 = "stablehlo.while"(%32, %5#0, %34, %36, %5#1, %38, %40) ({
    ^bb0(%arg40: tensor<20x20xi1>, %arg41: tensor<20x20xf32>, %arg42: tensor<20x20xf32>, %arg43: tensor<20x20xf32>, %arg44: tensor<20x20xf32>, %arg45: tensor<20x20xf32>, %arg46: tensor<20x20xf32>):
      %222 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %223 = "stablehlo.reduce"(%arg40, %222) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %224 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%224) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%223) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg33: tensor<20x20xi1>, %arg34: tensor<20x20xf32>, %arg35: tensor<20x20xf32>, %arg36: tensor<20x20xf32>, %arg37: tensor<20x20xf32>, %arg38: tensor<20x20xf32>, %arg39: tensor<20x20xf32>):
      %198 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %199 = "stablehlo.broadcast_in_dim"(%198) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %200 = "stablehlo.add"(%arg34, %199) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %201 = "stablehlo.divide"(%arg37, %200) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %202 = "stablehlo.multiply"(%arg38, %201) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %203 = "stablehlo.multiply"(%arg35, %arg37) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %204 = "stablehlo.multiply"(%200, %200) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %205 = "stablehlo.divide"(%203, %204) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %206 = "stablehlo.subtract"(%202, %205) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %207 = "stablehlo.add"(%arg39, %206) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %208 = "stablehlo.divide"(%arg37, %200) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %209 = "stablehlo.multiply"(%arg35, %208) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %210 = "stablehlo.add"(%arg36, %209) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %211 = "stablehlo.divide"(%209, %210) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %212 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %213 = "stablehlo.broadcast_in_dim"(%212) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %214 = "stablehlo.compare"(%211, %213) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %215 = "stablehlo.and"(%arg33, %214) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %216 = "stablehlo.select"(%arg33, %200, %arg34) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %217 = "stablehlo.select"(%arg33, %209, %arg35) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %218 = "stablehlo.select"(%arg33, %210, %arg36) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %219 = "stablehlo.select"(%arg33, %arg37, %arg37) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %220 = "stablehlo.select"(%arg33, %206, %arg38) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %221 = "stablehlo.select"(%arg33, %207, %arg39) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%215, %216, %217, %218, %219, %220, %221) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %42 = "stablehlo.multiply"(%41#3, %31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %43 = "stablehlo.divide"(%42, %5#0) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %44 = "stablehlo.not"(%18) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %45 = "stablehlo.and"(%30, %44) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %46 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %47 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %48 = "stablehlo.subtract"(%47, %5#0) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %49 = "stablehlo.add"(%5#1, %48) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %50 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %51 = "stablehlo.broadcast_in_dim"(%50) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %52 = "stablehlo.add"(%49, %51) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %53 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %55 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %56 = "stablehlo.broadcast_in_dim"(%55) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %57 = "stablehlo.add"(%5#1, %56) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %58 = "stablehlo.multiply"(%52, %5#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %59 = "stablehlo.divide"(%57, %58) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %60 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %61 = "stablehlo.broadcast_in_dim"(%60) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %62 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %63 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %64 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %65 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %66 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %67 = "stablehlo.broadcast_in_dim"(%66) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %68 = "stablehlo.negate"(%5#1) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.multiply"(%59, %68) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.subtract"(%67, %69) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %71 = "stablehlo.divide"(%70, %58) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %72 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %73:15 = "stablehlo.while"(%45, %59, %61, %48, %52, %72, %57, %58, %54, %5#1, %63, %65, %67, %68, %71) ({
    ^bb0(%arg16: tensor<20x20xi1>, %arg17: tensor<20x20xf32>, %arg18: tensor<20x20xf32>, %arg19: tensor<20x20xf32>, %arg20: tensor<20x20xf32>, %arg21: tensor<f32>, %arg22: tensor<20x20xf32>, %arg23: tensor<20x20xf32>, %arg24: tensor<20x20xf32>, %arg25: tensor<20x20xf32>, %arg26: tensor<20x20xf32>, %arg27: tensor<20x20xf32>, %arg28: tensor<20x20xf32>, %arg29: tensor<20x20xf32>, %arg30: tensor<20x20xf32>):
      %192 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f32>}> : () -> tensor<f32>
      %193 = "stablehlo.compare"(%arg21, %192) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %194 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %195 = "stablehlo.reduce"(%arg16, %194) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg31: tensor<i1>, %arg32: tensor<i1>):
        %197 = "stablehlo.or"(%arg31, %arg32) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%197) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %196 = "stablehlo.and"(%193, %195) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%196) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf32>, %arg3: tensor<20x20xf32>, %arg4: tensor<20x20xf32>, %arg5: tensor<20x20xf32>, %arg6: tensor<f32>, %arg7: tensor<20x20xf32>, %arg8: tensor<20x20xf32>, %arg9: tensor<20x20xf32>, %arg10: tensor<20x20xf32>, %arg11: tensor<20x20xf32>, %arg12: tensor<20x20xf32>, %arg13: tensor<20x20xf32>, %arg14: tensor<20x20xf32>, %arg15: tensor<20x20xf32>):
      %88 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %89 = "stablehlo.add"(%arg6, %88) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %90 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %91 = "stablehlo.broadcast_in_dim"(%90) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %92 = "stablehlo.add"(%arg4, %91) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %93 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %95 = "stablehlo.add"(%arg5, %94) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %96 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %97 = "stablehlo.multiply"(%92, %96) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %98 = "stablehlo.multiply"(%arg7, %95) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %99 = "stablehlo.multiply"(%arg9, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %100 = "stablehlo.subtract"(%98, %99) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %101 = "stablehlo.multiply"(%arg8, %95) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %102 = "stablehlo.multiply"(%arg10, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %103 = "stablehlo.subtract"(%101, %102) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %104 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %105 = "stablehlo.broadcast_in_dim"(%104) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %106 = "stablehlo.compare"(%103, %105) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %107 = "stablehlo.divide"(%100, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %108 = "stablehlo.subtract"(%arg2, %107) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %109 = "stablehlo.divide"(%108, %107) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %110 = "stablehlo.abs"(%109) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %111 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %112 = "stablehlo.broadcast_in_dim"(%111) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %113 = "stablehlo.select"(%106, %110, %112) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %114 = "stablehlo.select"(%106, %107, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %115 = "stablehlo.multiply"(%arg13, %95) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %116 = "stablehlo.subtract"(%115, %arg7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %117 = "stablehlo.multiply"(%arg11, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %118 = "stablehlo.subtract"(%116, %117) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %119 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %120 = "stablehlo.multiply"(%arg9, %119) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %121 = "stablehlo.add"(%118, %120) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %122 = "stablehlo.multiply"(%arg14, %95) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %123 = "stablehlo.subtract"(%122, %arg8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %124 = "stablehlo.multiply"(%arg12, %97) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %125 = "stablehlo.subtract"(%123, %124) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %126 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %127 = "stablehlo.multiply"(%arg10, %126) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %128 = "stablehlo.add"(%125, %127) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %129 = "stablehlo.multiply"(%114, %128) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %130 = "stablehlo.subtract"(%121, %129) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %131 = "stablehlo.divide"(%130, %103) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %132 = "stablehlo.select"(%106, %131, %arg15) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %133 = "stablehlo.subtract"(%132, %arg15) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %134 = "stablehlo.abs"(%133) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %135 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %136 = "stablehlo.broadcast_in_dim"(%135) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %137 = "stablehlo.select"(%106, %134, %136) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %138 = "stablehlo.abs"(%100) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %139 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %140 = "func.call"(%139) <{callee = @integer_pow}> : (tensor<f32>) -> tensor<f32>
      %141 = "stablehlo.broadcast_in_dim"(%140) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %142 = "stablehlo.compare"(%138, %141) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %143 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %144 = "stablehlo.broadcast_in_dim"(%143) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %145 = "stablehlo.multiply"(%arg7, %144) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %146 = "stablehlo.select"(%142, %145, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %147 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %148 = "stablehlo.broadcast_in_dim"(%147) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %149 = "stablehlo.multiply"(%100, %148) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %150 = "stablehlo.select"(%142, %149, %100) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %151 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %152 = "stablehlo.broadcast_in_dim"(%151) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %153 = "stablehlo.multiply"(%arg8, %152) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %154 = "stablehlo.select"(%142, %153, %arg8) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %155 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %156 = "stablehlo.broadcast_in_dim"(%155) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %157 = "stablehlo.multiply"(%103, %156) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %158 = "stablehlo.select"(%142, %157, %103) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %159 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %160 = "stablehlo.broadcast_in_dim"(%159) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %161 = "stablehlo.multiply"(%arg13, %160) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %162 = "stablehlo.select"(%142, %161, %arg13) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %163 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %164 = "stablehlo.broadcast_in_dim"(%163) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %165 = "stablehlo.multiply"(%arg14, %164) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %166 = "stablehlo.select"(%142, %165, %arg14) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %167 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %168 = "stablehlo.broadcast_in_dim"(%167) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %169 = "stablehlo.multiply"(%121, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %170 = "stablehlo.select"(%142, %169, %121) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %171 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %172 = "stablehlo.broadcast_in_dim"(%171) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %173 = "stablehlo.multiply"(%128, %172) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %174 = "stablehlo.select"(%142, %173, %128) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %175 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %176 = "stablehlo.broadcast_in_dim"(%175) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %177 = "stablehlo.compare"(%113, %176) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %178 = "stablehlo.and"(%arg1, %177) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %179 = "stablehlo.select"(%arg1, %114, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %180 = "stablehlo.select"(%arg1, %113, %arg3) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %181 = "stablehlo.select"(%arg1, %92, %arg4) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %182 = "stablehlo.select"(%arg1, %95, %arg5) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %183 = "stablehlo.select"(%arg1, %150, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %184 = "stablehlo.select"(%arg1, %158, %arg8) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %185 = "stablehlo.select"(%arg1, %146, %arg9) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %186 = "stablehlo.select"(%arg1, %154, %arg10) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %187 = "stablehlo.select"(%arg1, %162, %arg11) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %188 = "stablehlo.select"(%arg1, %166, %arg12) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %189 = "stablehlo.select"(%arg1, %170, %arg13) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %190 = "stablehlo.select"(%arg1, %174, %arg14) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %191 = "stablehlo.select"(%arg1, %132, %arg15) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%178, %179, %180, %181, %182, %89, %183, %184, %185, %186, %187, %188, %189, %190, %191) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %74 = "stablehlo.multiply"(%73#1, %31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %75 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %76 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %77 = "stablehlo.subtract"(%76, %43) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %78 = "stablehlo.select"(%18, %77, %74) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %79 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %80 = "stablehlo.broadcast_in_dim"(%79) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %81 = "stablehlo.compare"(%5#1, %80) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %82 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %83 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %84 = "stablehlo.select"(%81, %83, %78) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %85 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %86 = "stablehlo.broadcast_in_dim"(%85) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %87 = "stablehlo.select"(%13, %86, %84) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%87, %6) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%87) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf32>, tensor<20x20xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0x0490303E114F64C0C7B2A3C07707A13FE659153EC5D911404433AA40A8A58FC04CBC30C0362861C0DBE927406298BDBF6D06AA3F37561940D53256BFE266ADC0BD65C2BF63FA19C05902C3BFA0201C40B4AD1BC054B1B5C03FC2BB4059398A40AF818E3F5DCEA04028DB4F4009F2EC3F4DCA36C07825BBBF775DA3C0F76A343F0C8DDFBF2755C6BF73CE3C4079ED863E75983DC088C791BF66371D3F877B8440D66F02C0CB0E44C035CD0F403E07C73FAA2374C0274F14BEDAC0BE40AD0A39409E008240640F6EC02831A740B6DF9F3FC887A3BEC3350C4002EDD4C081E20E4030F96540D250D83E584E5BC06651FDBE0A0D79BFE139FD3E8CB3163DD428E73F979D983F2DA4D93F715DC53FE22FAD409A4C0A40B00D68C0BFE381C0F72FA0400821CB3E6E8011BF0DE6553B9837C23F9D1C97406DA9CBBFB09692BFAB1D954057CF16BE1ED7FEBF309803408C0576C066698DC0905D6BBFF25CD9BFE303A0BED77DF63EDCA10FBF44860EC017AB383FD4C570408A71E5BF81F004400CED273FAAD757BE896E1BC0832099BBC1F8F53E093CBD3F414429C05834E53F9F8464406C52923F0C62AABF3536CD3E26D98BBF58F6B43FCB3E3ABF23D859BF86EF83C056820D40F995A13FBEC74BC032D59A3FECED6E409558983E39C72F40701392BE7904143FF800E0BE4EBA88BF39DE82BFBD008140D6FC913F102702C01CEEB6C02ACE8340208745BE0D8B3AC00662404044D87E3F0735D3C0E05EE23F6AF696BF8728EA40124F2ABFAFEFDDBFF5B12A40B6B9FB3FEACEFBBED02C32C0E456F23F17B8263F9B7278C0ECAC69BF26334FBFD920C440F7CE95C0548894402821B2C0B5AB1E40B3AA44BE41345C3E10DCAFBEF3E01E4008558ABE51EBA1BF23A4DFC008DCBFBFC47863C09CEB9A40FE14AABF820078C0C10A02C069866F3FC610BD401C442340E49E12C0ADE8AAC078A253C055ACBD3F199FF2BF30CD4340F15E90401B7382BF18E798C04106ABBEA832A3C0E6D903C0AE3DB73F1DCA933F3292853F7EE2FA3EE61A20408D54B0BFCA444C3E87D500C0F0C282BF61BB81C03D8620BFF4C1BDBE162C604091E70FC06F37B8C0879E063F3C2BF23E16E51B3FD54B2540E0B03040D31694BF9F24433FA4F7EF3FCBCBEBBFE87BEE3E509D08C06BB99FC032F9AC3F784756409670933F971976BE3B704A406B1EC4BED1F09B40ADC5A5BF0C7FF3C03949BB4053334C403B554640E09EA03F0CE9AFBF19C4933FEA316B40BAF79C3F2E8E5540506EBEC09084043F5AA054BE75763CBF842D39BF855C9ABFECB6A63F5EBD0E3F487538C0E02189C0B03E6140536636C00D78DDC06B6BF33DDC5C51C098962340F8438F40A489F3C0E47FA0BF07220E403D9A79C000F59E3FD0EC92400B666AC061368B40E2CDEB3F96389BC04D0410BFC5AA2E3FC8F886BE4C359CC0BE7D7240F954A8BEA11354C0B518A53F225AC7C0E7BDC83F69FBFE3E6C6803C050C6A040CA537ABF30209AC029A3733FEBC50BC083CE1440CA0999BFC8B4DF3FCF6FA8C0B57B8DBE5A37154005759C3F5BA6913F81C36840893D8FBEB2182840E77296BFFC86EB3FD9B024408E61FBBFCA7373C008872C3EC2683040A1B59A3FFE90503D87C6224034B301BEEDD75ABFED9A88BF3FEF994052F0C33FC0ABB6C04CC46DBF3EE9E4BF5B3717405C4616BFECEFA93EBC8061BFD0A57B408F9B2F3F48BEA0BF1CA25A4036E559BF837708C02F9CD83E94BEA9BF86E9A440ECC168C01BB7333F7615EEC0691EAC3F12737B40CFA8A33E3ACFCC3F0C0302400BE2ABC04830224004EC973E30689ABF6E6E0BBF033081BFCEA920BF9AB019BFFF1C3B3FCF4AF7BD887AAEBE4B2935C0475691C0A16E4CBEA5A0BB405266C33F9797C43F15E64740DD1D6CC0E8D7E5BF7A404C40064B9F3F434CBBBF08190C408EC599404DAB3CBF9FF0223F82E162C0CEFC83C03AD662402823773D26C6AAC0873A9140DA8020407FA60F40B0F9BFBF76D60140AD16F13FA3AD4AC0975F794085FB45C08C4F2840306F49C0146E2CC00DC68FC0CB53CB3EF9D60C403B4741405603C43F112F6BBF43084CBD68C31B4011CBD93F264D324037A966C068D6B63EB13AB9BF09B2C83F42CA6DC0F8DB1CC0D04024C0E60E65C02B8FBCBF3D7059C0D8ECF5BFC102CFBE328AB3BF8D4BFFBE0799643F86ED7640FC8DD8BEA291803F6976E83F8A1A8D40A891464034C61340E0F89440B81A063F9C18B6BF4FFB8BC0"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    %4 = "stablehlo.constant"() <{value = dense<"0x58B47D406489FDBFCBAAA1BF99AA86C02A4E9DBF10EB1C40F43126403AB2C63F4A2C2FC0AEB825C09A87A5C0A3DABEBFA4C447408BA0AB404120CFBF6FE78A404AD689C0C34A09407DDF1F40DFA5E5BF8819B63DFB993240DE45AA403B09F5BF5C36AF3F79063F40180CD43F96B51CBF1758F13F428F0F40019E3FC0BE5803407AA69BBF5D63CE40055DFABF6CFA0F3E80B148408D60E440A2B0D4BEDAC262BFA50FD1BF75808CC00E6637C06EE67B3F612B9F40CD92FF3FEB298EC0515195C0D355453FECC786C0B2FF4240B166B3BF9DD3BDBFF9F906C0098A3BC02F73B9C0899A2540DB17673F81F451BF3BA89CBD9AB901C04A7F4C4097AAB8C0FA4AB73F245F6640EB0EF83E9B8D1A40F3C18BC0A50688BF53DD2BBF854C9D3F35023F3E23A4F6BD7FA201C01FC2F53F733D1740C7B5BB3F748480BFD7B2643FEACA85BFDA4F31C0BB2FA4C05A02F83E9944203FAA9D023F2A64AF3E4BAD86BFC024BB3F2CE7F5BFE76AA9BF2E750FBFAFF9FFBFFE51893CDB38F03F8CB485C01972D5BEB418A83F76E08740446AC4BFDC5595C0639DA63F7C931C4018341FBFABC6A5406221823DD24702C1EF5CC7BD96AF653F49919E40145180C04A1DCEBE2518823F28095FC0713A74BF0A0780C02EA6BEC059AA3CC0C1787C3E65FACF3E8D3848C062182BC09E6181BF411F53BFDDAD79BFBFED2A400906943F6A39B7BFE322A9BFC0219EBF02C1E0BE32778EC07E33C63F86DEEF3F401AE63E71B41AC0A74EBA40FA0B1CC0086319BF2748604080A0B4404C9C72C01A5386BF32C205C0F14108C0C55B823FA5782DC0F4E809C051E2E4BF4F7458405973CDBFD81E57C0C56DAFC09F4B5F40D04149BF4EBE6B3EE71EC5BF698295C0FDDBA5C0D83990C033BB3ABF20580E40D8CC0FC05CE73FC0E4636DC073FBFC3E9CD4E0BF112E1740C34957408189813FD2C33E40CB148EC07DAF0BBD4EF546C0CC4D81BF5968EBBF32276DBF57CD56C0C3A2C23FC17913C062A8DB3ED9E60FC05142F3BF894CF5BE07DF863F48025AC00C5882BE30E7DBBFDFB88140DF90CEBED22508BD0A8CB7BF7B839A3EB66017C00C855940C51F26BF223DB43E755B2B40010EACC06D61B8BF80E2EC3E79A061BF39C760C077B0D8BEE23E69BF36E39440E33E9BBF08AE1BC0FE4A5F40AFC278C03FD9133F7562BBBF2681C73F9B2B3BC048B3BA3F60F5BA3FC011BEBF7E7A4DC02ACCB9BE2EC4FCBF1626E1C0951A7C40F06E58C0375814C019EE22404C805C3FB87D6B40A2670FBE028D3CC040942FBF45260C407B0ACEC064C4F93FC8CCB2BF3D44CEBF1923123F90D0F2BFEFF78AC0982C87C067B851BE7B28823F16DE7640C61FDBBFE932F9BFCF0855C0BF06403D859D7B40492816BF7D478C40EE7B16C0200E0340CD8EDFC0768F5740AA1280BF579EBC3F6C4696BF97AC03C0933F4BBFB158103FDCE8E6BFDCF19C400759924018F3BFBF30F9B2409B83E93F471A1640F2C5EB3F07227D3FF080ED3FD2DDDC3DB4704A40EE3E3FC094E6853F64169D3DC8265A4010E5EE3EA923A43F2069CE3FA16DF63FCEB29ABF5BC80B408E299540D729A53F6CA0B9BF2BB93240A0FC81C0174BB63F1755DF3F462C15C02C03B5BF9C1D7BC0B3CDC73DFEC2B9BF760CBEBF6F034B408E89C14040508A3DEAFCF9BF3584943FC971E23E74F2FEC0E7184AC065B41CC0D61826405518B03F5509D5C0441B23C029B85EC03CBECEBEE3DDDB3F406A69400D2F3940977B99BF6CFC49BF2844BDBF744962406047F9BF603F1A40CE62C5BFEBB18CC0EF5A974007964240D5A5B8BD16D6E1BF956EA03DB01F0B404809CCBE558386BFBDD325BFA0FABD40224D69BFAC9B54BF31361FC0853C30403CE4B8C0D4D61740E9642AC035F0D4BF7371D5BFEDA39CBF8CAF5CC08B869C3F8C0346408CCBC6BFC9467FBF905E5EC0F3586FC0CFAE90C096DD59C0CB460F407C9339C065EFB44030532D403F0501C0CF2239C0D01F3240B7D474C05ACF1A40E776003FE89289C07420A73E595F17C07C4CB5BFD926F83F896A1AC07EEDDCBFFA8381402D6A973F00C68740BE320A40E1843A3FA0FBE6BE349C92BF80D1913F794F15BE7E1549BFE90157BEF3853E3FC2F1D640FA5C09C0979CFB3E044F03C014350340A42321C05765443FDF14463FA67136C054FD2040F78A48C00EBB1D407CF18E3EEE1AD340E8060CC0E470E73F119D2E40CC15A6BFD0EA09C0C5A535C00D8E7340098039C00997EF3F"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%3, %4) : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x5B1E7C3A0000803F0000803F0000803F0000803F0332BD3EE54C683F0000803F0000803F0000803F0000803F0000803FC002A03D1C694E3D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F9490093F0000803F90BD963EE234523F70A1503F0000803F0000803F0000803F0000803FAAF5943D0000803F0000803F0000803F2EB1B73E0000803F0000803F0000803F0000803F0000803F0000803F0000803FDA6F193F0000803F0000803F0000803F0000803F632B7E3F0000803F8112563F0000803F0000803F0000803F0000803F0000803F12A2283F58BC153E0000803F0000803F0000803F7293383C0000803F74A6043F9DA9223DC61C5C3F54A8463E0000803F0000803F0000803F0000803FE5FF7F3F0000803F0000803FD4FB3B3958A5493E44A8793F0000803F0000803F0000803F0000803F0000803F771E6C3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FC0E7E53E0000803F0000803F5CAF363E1516763F0000803F0000803F0000803FE3D0873C0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FF2E79B3E735A7C3F0000803F0000803F0000803F0000803F0000803FACEF393F24B9BD3E0000803F0000803F0000803F0000803F0000803FA8374C3FBB121C3E0000803F0000803F0000803F0000803F0000803F0000803FD2FC643D0000803F0000803F0000803F0000803FC2AB5A3E0000803F0000803F0000803FD477623F0000803F0000803F0000803F4AA15F3E0000803FF8A06E3E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803FF1C4AB3D94BE5D3F9EDA5A3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F6DDDBB3E0000803F0000803F0000803F0B258A3A0000803F0000803F0000803F0000803F0000803F42D7E63E0000803F0000803F14A3B73C0000803F0000803F0518793F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F5A387D3F0000803F0000803F0000803F0000803F6FE47A3F0000803F0000803F0000803F0000803F0000803F5ADB033D0000803F0000803FB0862F3FA8E4053FCC50B73E0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F5683E83C0000803F0000803F0000803F0000803F0000803F3920FE3D0000803F94ACA83C0000803F0000803F0000803F8212013E0000803F0000803F0000803F0000803F0000803F13EE7E3F0000803F0000803F904F9A3C0000803F56A6473C3A84643D0000803FAC29763F0000803F0000803FFA3C623F0000803F0000803F0000803FB24A7E3F0000803F0000803F58E5393F2516893E11C1373E0000803F0000803FA311E23D0000803F0000803FDEADBB3E0000803F0000803F19CF713C0000803F0000803F0000803FE6D07F3F0000803F0000803F0000803F0172803EE7137D3F0000803F0000803F0000803F0000803F0000803F0000803F0000803F9FC8713F0000803F0000803F0000803F0000803F0000803FC9E7AA3B0000803F0000803F0000803F0000803F0000803F0000803FD3B3433F0000803F0000803FBB02583D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F3FAFA43A0000803F0000803F0000803F0000803F0000803F55ED753F0000803F0000803F0000803F0000803F0000803FCD4A663F77AF8F3D0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0FCF1C380000803F0000803F0000803FF241943E0000803FC79C9F3ED5D7633F0000803F9DE07F3F0000803F0000803F0000803F0000803F0000803F5804443B9CAB393FF6DA543EBAC4723E0000803F0000803F0000803F6ED0163F0000803F0000803F0000803F0000803F7CE38D3B0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F8E7D8E3D4EEC7F3F0000803F0000803FE2ACCE3E5A64473F0000803F0000803F0000803FAC59CE3B0000803F0000803F"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%2) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


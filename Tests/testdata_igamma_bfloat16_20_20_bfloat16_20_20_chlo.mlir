"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xbf16>, tensor<20x20xbf16>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xbf16>
    %7 = "stablehlo.convert"(%5#0) : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %8 = "stablehlo.convert"(%5#1) : (tensor<20x20xbf16>) -> tensor<20x20xf32>
    %9 = "stablehlo.compare"(%7, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %10 = "stablehlo.compare"(%8, %8) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %11 = "stablehlo.or"(%9, %10) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %12 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %13 = "stablehlo.broadcast_in_dim"(%12) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %14 = "stablehlo.compare"(%8, %13) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %15 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<f32>}> : () -> tensor<f32>
    %16 = "stablehlo.broadcast_in_dim"(%15) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %17 = "stablehlo.compare"(%8, %16) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %18 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %19 = "stablehlo.broadcast_in_dim"(%18) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %20 = "stablehlo.compare"(%8, %19) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %21 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %23 = "stablehlo.compare"(%7, %22) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %24 = "stablehlo.or"(%20, %23) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %25 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %26 = "stablehlo.broadcast_in_dim"(%25) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %27 = "stablehlo.compare"(%8, %26) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %28 = "stablehlo.compare"(%8, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %29 = "stablehlo.and"(%27, %28) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %30 = "stablehlo.log"(%8) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %31 = "stablehlo.multiply"(%7, %30) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %32 = "stablehlo.subtract"(%31, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %33 = "chlo.lgamma"(%7) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %34 = "stablehlo.subtract"(%32, %33) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %35 = "stablehlo.constant"() <{value = dense<3.40282347E+38> : tensor<f32>}> : () -> tensor<f32>
    %36 = "stablehlo.log"(%35) : (tensor<f32>) -> tensor<f32>
    %37 = "stablehlo.negate"(%36) : (tensor<f32>) -> tensor<f32>
    %38 = "stablehlo.broadcast_in_dim"(%37) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %39 = "stablehlo.compare"(%34, %38) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %40 = "stablehlo.exponential"(%34) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %41 = "stablehlo.or"(%14, %24) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %42 = "stablehlo.or"(%41, %39) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %43 = "stablehlo.or"(%42, %11) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %44 = "stablehlo.not"(%43) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %45 = "stablehlo.and"(%44, %29) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %46 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %47 = "stablehlo.broadcast_in_dim"(%46) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %48 = "stablehlo.subtract"(%47, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %49 = "stablehlo.add"(%8, %48) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %50 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %51 = "stablehlo.broadcast_in_dim"(%50) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %52 = "stablehlo.add"(%49, %51) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %53 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %54 = "stablehlo.broadcast_in_dim"(%53) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %55 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %56 = "stablehlo.broadcast_in_dim"(%55) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %57 = "stablehlo.add"(%8, %56) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %58 = "stablehlo.multiply"(%52, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %59 = "stablehlo.divide"(%57, %58) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %60 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %61 = "stablehlo.broadcast_in_dim"(%60) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %62 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %63 = "stablehlo.broadcast_in_dim"(%62) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %64 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %65 = "stablehlo.broadcast_in_dim"(%64) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %66 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %67 = "stablehlo.broadcast_in_dim"(%66) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %68 = "stablehlo.negate"(%8) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %69 = "stablehlo.multiply"(%59, %68) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %70 = "stablehlo.subtract"(%67, %69) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %71 = "stablehlo.divide"(%70, %58) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %72 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %73:15 = "stablehlo.while"(%45, %59, %61, %48, %52, %72, %57, %58, %54, %8, %63, %65, %67, %68, %71) ({
    ^bb0(%arg32: tensor<20x20xi1>, %arg33: tensor<20x20xf32>, %arg34: tensor<20x20xf32>, %arg35: tensor<20x20xf32>, %arg36: tensor<20x20xf32>, %arg37: tensor<f32>, %arg38: tensor<20x20xf32>, %arg39: tensor<20x20xf32>, %arg40: tensor<20x20xf32>, %arg41: tensor<20x20xf32>, %arg42: tensor<20x20xf32>, %arg43: tensor<20x20xf32>, %arg44: tensor<20x20xf32>, %arg45: tensor<20x20xf32>, %arg46: tensor<20x20xf32>):
      %234 = "stablehlo.constant"() <{value = dense<2.000000e+03> : tensor<f32>}> : () -> tensor<f32>
      %235 = "stablehlo.compare"(%arg37, %234) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %236 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %237 = "stablehlo.reduce"(%arg32, %236) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg47: tensor<i1>, %arg48: tensor<i1>):
        %239 = "stablehlo.or"(%arg47, %arg48) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%239) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      %238 = "stablehlo.and"(%235, %237) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%238) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg17: tensor<20x20xi1>, %arg18: tensor<20x20xf32>, %arg19: tensor<20x20xf32>, %arg20: tensor<20x20xf32>, %arg21: tensor<20x20xf32>, %arg22: tensor<f32>, %arg23: tensor<20x20xf32>, %arg24: tensor<20x20xf32>, %arg25: tensor<20x20xf32>, %arg26: tensor<20x20xf32>, %arg27: tensor<20x20xf32>, %arg28: tensor<20x20xf32>, %arg29: tensor<20x20xf32>, %arg30: tensor<20x20xf32>, %arg31: tensor<20x20xf32>):
      %130 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %131 = "stablehlo.add"(%arg22, %130) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %132 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %133 = "stablehlo.broadcast_in_dim"(%132) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %134 = "stablehlo.add"(%arg20, %133) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %135 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %136 = "stablehlo.broadcast_in_dim"(%135) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %137 = "stablehlo.add"(%arg21, %136) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %138 = "stablehlo.broadcast_in_dim"(%131) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %139 = "stablehlo.multiply"(%134, %138) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %140 = "stablehlo.multiply"(%arg23, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %141 = "stablehlo.multiply"(%arg25, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %142 = "stablehlo.subtract"(%140, %141) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %143 = "stablehlo.multiply"(%arg24, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %144 = "stablehlo.multiply"(%arg26, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %145 = "stablehlo.subtract"(%143, %144) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %146 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %147 = "stablehlo.broadcast_in_dim"(%146) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %148 = "stablehlo.compare"(%145, %147) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %149 = "stablehlo.divide"(%142, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %150 = "stablehlo.subtract"(%arg18, %149) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %151 = "stablehlo.divide"(%150, %149) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %152 = "stablehlo.abs"(%151) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %153 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %154 = "stablehlo.broadcast_in_dim"(%153) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %155 = "stablehlo.select"(%148, %152, %154) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %156 = "stablehlo.select"(%148, %149, %arg18) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %157 = "stablehlo.multiply"(%arg29, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %158 = "stablehlo.subtract"(%157, %arg23) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %159 = "stablehlo.multiply"(%arg27, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %160 = "stablehlo.subtract"(%158, %159) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %161 = "stablehlo.broadcast_in_dim"(%131) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %162 = "stablehlo.multiply"(%arg25, %161) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %163 = "stablehlo.add"(%160, %162) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %164 = "stablehlo.multiply"(%arg30, %137) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %165 = "stablehlo.subtract"(%164, %arg24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %166 = "stablehlo.multiply"(%arg28, %139) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %167 = "stablehlo.subtract"(%165, %166) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %168 = "stablehlo.broadcast_in_dim"(%131) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %169 = "stablehlo.multiply"(%arg26, %168) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %170 = "stablehlo.add"(%167, %169) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %171 = "stablehlo.multiply"(%156, %170) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %172 = "stablehlo.subtract"(%163, %171) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %173 = "stablehlo.divide"(%172, %145) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %174 = "stablehlo.select"(%148, %173, %arg31) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %175 = "stablehlo.subtract"(%174, %arg31) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %176 = "stablehlo.abs"(%175) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %177 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %178 = "stablehlo.broadcast_in_dim"(%177) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %179 = "stablehlo.select"(%148, %176, %178) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %180 = "stablehlo.abs"(%142) : (tensor<20x20xf32>) -> tensor<20x20xf32>
      %181 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %182 = "func.call"(%181) <{callee = @integer_pow}> : (tensor<f32>) -> tensor<f32>
      %183 = "stablehlo.broadcast_in_dim"(%182) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %184 = "stablehlo.compare"(%180, %183) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %185 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %186 = "stablehlo.broadcast_in_dim"(%185) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %187 = "stablehlo.multiply"(%arg23, %186) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %188 = "stablehlo.select"(%184, %187, %arg23) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %189 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %190 = "stablehlo.broadcast_in_dim"(%189) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %191 = "stablehlo.multiply"(%142, %190) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %192 = "stablehlo.select"(%184, %191, %142) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %193 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %194 = "stablehlo.broadcast_in_dim"(%193) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %195 = "stablehlo.multiply"(%arg24, %194) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %196 = "stablehlo.select"(%184, %195, %arg24) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %197 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %198 = "stablehlo.broadcast_in_dim"(%197) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %199 = "stablehlo.multiply"(%145, %198) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %200 = "stablehlo.select"(%184, %199, %145) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %201 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %202 = "stablehlo.broadcast_in_dim"(%201) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %203 = "stablehlo.multiply"(%arg29, %202) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %204 = "stablehlo.select"(%184, %203, %arg29) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %205 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %206 = "stablehlo.broadcast_in_dim"(%205) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %207 = "stablehlo.multiply"(%arg30, %206) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %208 = "stablehlo.select"(%184, %207, %arg30) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %209 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %210 = "stablehlo.broadcast_in_dim"(%209) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %211 = "stablehlo.multiply"(%163, %210) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %212 = "stablehlo.select"(%184, %211, %163) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %213 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %214 = "stablehlo.broadcast_in_dim"(%213) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %215 = "stablehlo.multiply"(%170, %214) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %216 = "stablehlo.select"(%184, %215, %170) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %217 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %218 = "stablehlo.broadcast_in_dim"(%217) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %219 = "stablehlo.compare"(%155, %218) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %220 = "stablehlo.and"(%arg17, %219) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %221 = "stablehlo.select"(%arg17, %156, %arg18) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %222 = "stablehlo.select"(%arg17, %155, %arg19) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %223 = "stablehlo.select"(%arg17, %134, %arg20) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %224 = "stablehlo.select"(%arg17, %137, %arg21) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %225 = "stablehlo.select"(%arg17, %192, %arg23) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %226 = "stablehlo.select"(%arg17, %200, %arg24) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %227 = "stablehlo.select"(%arg17, %188, %arg25) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %228 = "stablehlo.select"(%arg17, %196, %arg26) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %229 = "stablehlo.select"(%arg17, %204, %arg27) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %230 = "stablehlo.select"(%arg17, %208, %arg28) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %231 = "stablehlo.select"(%arg17, %212, %arg29) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %232 = "stablehlo.select"(%arg17, %216, %arg30) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %233 = "stablehlo.select"(%arg17, %174, %arg31) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%220, %221, %222, %223, %224, %131, %225, %226, %227, %228, %229, %230, %231, %232, %233) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<f32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %74 = "stablehlo.multiply"(%73#1, %40) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %75 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %76 = "stablehlo.broadcast_in_dim"(%75) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %77 = "stablehlo.subtract"(%76, %74) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %78 = "stablehlo.not"(%29) : (tensor<20x20xi1>) -> tensor<20x20xi1>
    %79 = "stablehlo.and"(%44, %78) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %80 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %81 = "stablehlo.broadcast_in_dim"(%80) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %82 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %83 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %84 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %85 = "stablehlo.broadcast_in_dim"(%84) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %86 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %87 = "stablehlo.broadcast_in_dim"(%86) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %88:7 = "stablehlo.while"(%79, %7, %81, %83, %8, %85, %87) ({
    ^bb0(%arg8: tensor<20x20xi1>, %arg9: tensor<20x20xf32>, %arg10: tensor<20x20xf32>, %arg11: tensor<20x20xf32>, %arg12: tensor<20x20xf32>, %arg13: tensor<20x20xf32>, %arg14: tensor<20x20xf32>):
      %127 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %128 = "stablehlo.reduce"(%arg8, %127) <{dimensions = array<i64: 0, 1>}> ({
      ^bb0(%arg15: tensor<i1>, %arg16: tensor<i1>):
        %129 = "stablehlo.or"(%arg15, %arg16) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%129) : (tensor<i1>) -> ()
      }) : (tensor<20x20xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%128) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<20x20xi1>, %arg2: tensor<20x20xf32>, %arg3: tensor<20x20xf32>, %arg4: tensor<20x20xf32>, %arg5: tensor<20x20xf32>, %arg6: tensor<20x20xf32>, %arg7: tensor<20x20xf32>):
      %103 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
      %104 = "stablehlo.broadcast_in_dim"(%103) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %105 = "stablehlo.add"(%arg2, %104) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %106 = "stablehlo.divide"(%arg5, %105) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %107 = "stablehlo.multiply"(%arg6, %106) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %108 = "stablehlo.multiply"(%arg3, %arg5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %109 = "stablehlo.multiply"(%105, %105) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %110 = "stablehlo.divide"(%108, %109) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %111 = "stablehlo.subtract"(%107, %110) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %112 = "stablehlo.add"(%arg7, %111) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %113 = "stablehlo.divide"(%arg5, %105) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %114 = "stablehlo.multiply"(%arg3, %113) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %115 = "stablehlo.add"(%arg4, %114) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %116 = "stablehlo.divide"(%114, %115) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %117 = "stablehlo.constant"() <{value = dense<1.1920929E-7> : tensor<f32>}> : () -> tensor<f32>
      %118 = "stablehlo.broadcast_in_dim"(%117) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
      %119 = "stablehlo.compare"(%116, %118) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
      %120 = "stablehlo.and"(%arg1, %119) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
      %121 = "stablehlo.select"(%arg1, %105, %arg2) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %122 = "stablehlo.select"(%arg1, %114, %arg3) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %123 = "stablehlo.select"(%arg1, %115, %arg4) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %124 = "stablehlo.select"(%arg1, %arg5, %arg5) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %125 = "stablehlo.select"(%arg1, %111, %arg6) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      %126 = "stablehlo.select"(%arg1, %112, %arg7) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
      "stablehlo.return"(%120, %121, %122, %123, %124, %125, %126) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    }) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>) -> (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>, tensor<20x20xf32>)
    %89 = "stablehlo.multiply"(%88#3, %40) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %90 = "stablehlo.divide"(%89, %7) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %91 = "stablehlo.select"(%29, %77, %90) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %92 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %93 = "stablehlo.broadcast_in_dim"(%92) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %94 = "stablehlo.select"(%14, %93, %91) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %95 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %96 = "stablehlo.broadcast_in_dim"(%95) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %97 = "stablehlo.select"(%17, %96, %94) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %98 = "stablehlo.or"(%24, %11) : (tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %99 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<f32>}> : () -> tensor<f32>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %101 = "stablehlo.select"(%98, %100, %97) : (tensor<20x20xi1>, tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %102 = "stablehlo.convert"(%101) : (tensor<20x20xf32>) -> tensor<20x20xbf16>
    "stablehlo.custom_call"(%102, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    "func.return"(%102) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xbf16>, tensor<20x20xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0x98C05D4094C05B3F3CC0C0BFF83F42C012C039402DBE5AC0D0BFDDBF05BFCA3F0E3EEB3D9A40D540BEC0CABF1EBF4D40793F8C40BEBFAABF13BF1D3F15C04A4089BF8B409CBF85BFB5C0D63ED43F5940A6BFE63EA040A3BF4D3FEEBF23C0A14076400340E03F53C07DC050C0C43EB6408FBD1040CF3F3AC0524027402AC00440813F773FE2BE23C023C0633E81BFD83F9BC06FBF67C0553F67400FBF964038C0A940E53F0140FFBFAEC02BC099408440034002C0D1BFD9408AC087C0E23F793F9740553E4ABF9E3F9CC0B440024026C02BC04CC0ABBE2540D7BFAA3F20C02A4042C0FA3EDC3E95404CBF58C062C0854056BF10BE28BE8B40BA3F663F9AC0993F17405F3F77BFE33E9A3FE5BE5540B6BE0AC07EC022BF1240E0C00E3DD9C0A84007BFAD3EB3C099BFFC3F3E40114004BE2EC058BF3840B03FD74086C0ACC0AE400EBF5940BFBF3B40B83F54C0B2BF80C098BC2AC0B3BFEB3F9F3F0B40A23E94BFBF401EC02F4066BF65404AC0C33FAEC059C066BF75C0F03F3C40B53F59406740D8C069C0843F18C039BFABBF3D40F93FA6C031C036C02C40DD3D3AC074408CC03FC0DC3FB4BF3C3DF43F2FBF353FC6BF793F9D40C83F7CBF84C06F3F96C08BC011C0A3C05240324058C090BF0EC0F8C0D2BF393E11C0DB3F274045C0963E25C06D3F7DC0F9BF8AC07940E53FB5C0C340B4BFF5BFBF40323EB83F1E3F1940AD40EFBF843D09C143C02BC019C0EABFD5BF3340A1BFF13FCABF7F4004C075BD0941143F95C07FC087C0A9C0853F524098BF04C07B407F3F3740B1C0A3BF06402C409F3F79402D4028C00AC0E3BFFA3D11404C3F86C024C07140913F20C02340833F7F4038C09840E43F453D593F4D3F34BF81BFBCBF9F40FC40A0C01D406A4034408940FFC0B2BEED3F38406DC09A407E408A40A4C0213FA53E5A40D5BFDB3F573E23C019C020C0103FE9BE41405140A63E754099406C40B2BEAE3E683FC03FBB3E8F400640A7C06AC087BF3F3F74BFD63F053F63BF19C02D3F6DBF69C0C1BFCEBE2DC085BE723D0EC0544037C0EFBF8EC020C0B3BF1BC004C06CC0D3C0AFBF2A4077405EC0D740AFC043C0F7BF0BC027C087BFD5BF27404EC0754042C0203FB63F"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    %4 = "stablehlo.constant"() <{value = dense<"0xA94084405640674081405B3F72C000BF63C01DC076C0AB3ECF3FD640214088401BC0F53F5D40BBBF3540AA405B3FB3406AC096BFCE3F28403FC0ED406D401BC0C6BFF4BF883F933FACBEBD3F923F33C088C0F3BF273FCD40ED3E923F04409D3FA83E9EC0633F53C004C009BFFC3F883F08BD87C00C3F6CBF404087407C4025C0D9BFCBC0A03FD43F3E403340EF3FC93EDABF334026BFAC3FB73DF8BE7140E44024BFD1BEA7BE504056C0AA405940443DEEBEA3C0B83F87BF58C028405D40B0C01640D03F983F914096C089BD124091BF3BC06FBF3940D1C01940DCBF90C0BABF61409E3FF0BFFFBF7C405640ADC036404E408EBF56C0CEBED23FE7BE62C02CC076C0484018C0C73F88C06740DA3F18C0004016C089BF14BE54C0DFC0AE40F5BF3140B0BF2BC04EC012BF5DBF904098C0BCC0AAC024C09F40CC3F8CC081BFD1404D407A3FC03EB63FE8408340AB3EBC40B040DCBF21C07ABF6B3F51BF92BF8EC0EABF8FC090401BBFA8C0A7C030C01A4075C0C4C02FC0794060C0B63F5DC0EDC0BF40ABBFE93F354000C0733FC5BE5B3D5ABFD1BF12406D3F13BF1EBEFABF504015BCADBEEABE063F603F12C0684025405FC0613F10BFBBBE8A3FE93F19C0D8BF2EBF7A3F05402E4043C029C00DC0F53FB23F8840F4BF034111C02F3F2340EB3F873F47BF66C08F3F2EC095BF8EBFEA3E94BF96404EC0123FFC3FDBBFA43FCABFB4BF2A408A40B8C094BFFC3F0D4085BF7EC0343F50C087C019C04A400BC0174050C05D40CBBF97C015C000416CC0404091C0CC3D7540C1BF544077C0DB3E87408FBFA43F2BC0624087C0B64060C0783F143F0640983EAEBED4BFECBF70C0F6BFDA3E1FC03ABF953F954022C09E40EB3F53C0E93FC6C0FBBFE34006BEFE3F97C0D9BF5A401141743E80BF823FB140BB403F4076C084C016C054BD5B3E943FE6BF27C057BF9F3E83C0FABE63403CC03E40BB3F69BE2AC0F73AEA4081C0394062BF1BC097C071BF053F97C08E3F663D3040633E3F408CC08ABF7B40FDBF773F963F34BF46BFB53F47BEB5409DBFBF40F2C0DFC0B13F81C01A408FBE73BEE03FDDBFD2BFD0BFD9BF42C0843F66C0D9BFC83F86C0D3BFD7405FBE95BF563D834068BF"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    "func.return"(%3, %4) : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0xC07F333FC07F7B3FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F753FC07F7E3F983EC07FC07FC07FC07F663FC07FC07FC07FC07FC07F803FC07FC07FC07FC07FC07FC07FC07F6F3FDB3EC07FC07FC07F173AC07FF43EC07FC07F043C133AC07F963EC07FC07FC07F783FBB3AC07FC07F403EC07F033F5A3FC07FC07FC07FC07FC07FC07FC07F7F3FC07FD93DC07FC07FC07F4C3F3D37C07FC43EC07FC07FC07FC07FC07FC07FC07F943EFF33C07FC07FC07FC07FC07FC07F653FC07FEC3D7A3FC07F7B3FC07FC07F283FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F633FC07FC07FC07FC07FC07F943EC07FC07FC07FC07F2A3FC07FC07FC07FC07F773FC07F6F3FC07FC07F3A3EC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F6A3FC07FC07FC07FC07F7C3F053BC07FC07F383FC07F353DC07F3E3E7F3FC07FC07FC07FC07FC07FC07FC07FFE3EC07FC07FC07FC07FC07F5D3FC07FC07FC07FC07FC07FC07FC07FC07F6A3FC07F1D3FC07FC07FC07FC07F543FC07FC07FC07FC07FDF3AC07FC07FC07FD23DC07FC07FC07FC07FC07FC07FC07F7A3F793EC07F7D3FC07FC07F253BC07FC07FC07F5B3FC07FC07FC07FC07F903E113FC07FC07FC07FC07FC07F803FC07F803FC07FC07F7D3FC07F2F3FC07FC07FC07FC07FC07FC07FE036C07FC07FC07F693F403FC07F843EC07FC07F803FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FB93AC07FC07FC07FC07FC07F803FC07FC07FC07FAB367A3FC07FC07FC07F6E3D573FC07F433DC07FC07FC07FC07FC07F413E0C3FC07FC07FC07FC07FC07FC07FC07F853AC07FC07FC93E803FC07F7F3FC07FC07FC07FC07FC07FC07FC07F433EC07FC07FC07FC07F0D3DC07FC07F2E3F573F943EC07FC07FC07FC07FC07FD43EC07FC07FC07FC07FC07FC07F2F3FC07F7E3F993DC07FC07FC07F803FC07F613FC07FC07FC07FC07FC07FC07F463FC07F543FF53EC07FC07FC07FC07FC07FC07FC07FC07FC07F7E3FC07F653FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07FC07F7A3FC07FC07FC07F7E3FC07F"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    "func.return"(%2) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<f32>) -> tensor<f32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<f32>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.divide"(%0, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%1) : (tensor<f32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()


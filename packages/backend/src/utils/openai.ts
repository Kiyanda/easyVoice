import { AxiosError } from 'axios'
import { MODEL_NAME, OPENAI_BASE_URL, OPENAI_API_KEY } from '../config'
import { logger } from './logger'
import { fetcher } from './request'

// 配置接口定义
interface OpenAIConfig {
  baseURL?: string
  model?: string
  timeout: number
  apiKey?: string
}

/**
 * 创建 OpenAI 客户端实例
 * @returns OpenAI 工具函数集合
 */
export function createOpenAIClient() {
  // 默认配置
  let currentConfig: OpenAIConfig = {
    baseURL: OPENAI_BASE_URL,
    model: MODEL_NAME,
    timeout: 120000, // 增加到 120 秒，适应不稳定网络
    apiKey: OPENAI_API_KEY,
  }
  logger.debug(`init openai with: `, {
    ...currentConfig,
    apiKey: currentConfig?.apiKey ? currentConfig?.apiKey?.slice(0, 10) + '***' : undefined,
  })
  // 设置 headers
  const getHeaders = () => ({
    Authorization: `Bearer ${currentConfig.apiKey}`,
    'Content-Type': 'application/json',
  })

  /**
   * 创建 Chat Completion（带重试机制）
   * @param request 请求参数
   * @param customConfig 自定义配置，可覆盖默认配置
   */
  async function createChatCompletion(
    request: ChatCompletionRequest,
    customConfig?: Partial<OpenAIConfig>
  ): Promise<ChatCompletionResponse> {
    const maxRetries = 5 // 增加到 5 次重试
    let lastError: Error | null = null

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const mergedConfig = {
          ...currentConfig,
          ...customConfig,
        }

        const response = await fetcher.post<ChatCompletionResponse>(
          `${mergedConfig.baseURL}${mergedConfig.baseURL?.endsWith('/') ? '' : '/'}chat/completions`,
          {
            model: request.model || mergedConfig.model,
            temperature: request.temperature ?? 1.0,
            max_tokens: request.max_tokens,
            top_p: request.top_p ?? 1.0,
            stream: request.stream ?? false,
            ...request,
          },
          {
            headers: getHeaders(),
            timeout: mergedConfig.timeout,
          }
        )

        // 验证响应数据
        const responseData = response.data as any
        if (!responseData) {
          logger.error('OpenAI API returned empty data', { response })
          throw new Error('Invalid API response: empty data')
        }

        // 检查是否是字符串而不是对象
        if (typeof responseData === 'string') {
          logger.error('OpenAI API returned string instead of object', {
            dataPreview: (responseData as string).slice(0, 200)
          })
          throw new Error('Invalid API response: received string instead of JSON object')
        }

        // 验证响应结构
        if (!responseData.choices || !Array.isArray(responseData.choices)) {
          logger.error('OpenAI API returned invalid response structure', {
            data: JSON.stringify(responseData).slice(0, 500)
          })
          throw new Error('Invalid API response: missing or invalid choices array')
        }

        // 验证choices数组不为空
        if (responseData.choices.length === 0) {
          logger.error('OpenAI API returned empty choices array', {
            data: JSON.stringify(responseData).slice(0, 500)
          })
          throw new Error('Invalid API response: empty choices array')
        }

        // 验证message content存在
        if (!responseData.choices[0].message) {
          logger.error('OpenAI API returned no message in choice', {
            choice: JSON.stringify(responseData.choices[0]).slice(0, 500)
          })
          throw new Error('Invalid API response: no message in choice')
        }

        if (!responseData.choices[0].message.content) {
          logger.error('OpenAI API returned empty message content', {
            message: JSON.stringify(responseData.choices[0].message).slice(0, 500),
            fullResponse: JSON.stringify(responseData).slice(0, 1000)
          })
          throw new Error('Invalid API response: empty message content')
        }

        // 记录成功的响应
        logger.debug('OpenAI API response received successfully', {
          contentLength: responseData.choices[0].message.content.length,
          contentPreview: responseData.choices[0].message.content.slice(0, 100)
        })

        return responseData as ChatCompletionResponse
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error))

        if (error instanceof AxiosError) {
          const errorCode = error.code
          const status = error.response?.status
          const errorData = error.response?.data
          const isNetworkError = ['ECONNRESET', 'ECONNABORTED', 'ETIMEDOUT', 'ENOTFOUND'].includes(errorCode || '')
          const isRateLimitError = status === 429 || (errorData && typeof errorData === 'object' && 'error' in errorData && errorData.error && typeof errorData.error === 'object' && 'type' in errorData.error && errorData.error.type === 'rate_limit_exceeded')
          const isServerError = status && status >= 500 && status < 600
          const shouldRetry = isNetworkError || isRateLimitError || isServerError

          logger.warn(`OpenAI API attempt ${attempt}/${maxRetries} failed`, {
            code: errorCode,
            status,
            message: error.message,
            isNetworkError,
            isRateLimitError,
            isServerError,
            errorData: errorData ? JSON.stringify(errorData).slice(0, 200) : undefined
          })

          // 如果是可重试的错误且还有重试机会，则继续重试
          if (shouldRetry && attempt < maxRetries) {
            // 根据错误类型调整延迟
            let delay: number
            if (isRateLimitError) {
              // 限流错误使用更长的延迟
              delay = Math.min(5000 * Math.pow(2, attempt - 1), 60000) // 5s, 10s, 20s, 40s, 60s
            } else {
              // 网络错误或服务器错误使用较短延迟
              delay = Math.min(2000 * Math.pow(1.5, attempt - 1), 15000)
            }
            logger.info(`Retrying in ${delay}ms... (attempt ${attempt + 1}/${maxRetries})`)
            await new Promise(resolve => setTimeout(resolve, delay))
            continue
          }
        }

        // 非网络错误或最后一次重试失败，直接抛出
        if (attempt >= maxRetries) {
          logger.error('OpenAI API failed after all retries', { error: lastError.message })
          throw new Error(
            `Chat completion request failed after ${maxRetries} attempts: ${lastError.message}`
          )
        }
      }
    }

    throw new Error(
      `Chat completion request failed: ${lastError?.message || 'Unknown error'}`
    )
  }

  /**
   * 获取可用模型列表
   */
  async function getModels(): Promise<{ data: { id: string }[] }> {
    try {
      const response = await fetcher.get<{ data: { id: string }[] }>(
        `${currentConfig.baseURL}/models`,
        {},
        {
          headers: getHeaders(),
          timeout: currentConfig.timeout,
        }
      )
      return response.data
    } catch (error) {
      throw new Error(
        `Get models failed: ${error instanceof Error ? error.message : String(error)}`
      )
    }
  }

  /**
   * 动态更新配置
   * @param newConfig 新的配置参数
   */
  function config(newConfig: Partial<OpenAIConfig>) {
    currentConfig = {
      ...currentConfig,
      ...newConfig,
    }
    logger.debug(`openai currentConfig:`, currentConfig)
  }

  return {
    createChatCompletion,
    getModels,
    config,
  }
}

export const openai = createOpenAIClient()
